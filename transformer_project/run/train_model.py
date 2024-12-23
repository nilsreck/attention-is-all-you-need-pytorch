import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm
from pathlib import Path

from transformer_project.modelling.transformer import Transformer
from transformer_project.data.translation_dataset import TranslationDataset
from transformer_project.modelling.lr_scheduler import LR_Scheduler
from transformer_project.preprocessing.clean_data import load_or_clean_data
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Transformer(
    vocab_size=50000,
    d_model=32,
    n_heads=8,
    num_decoder_layers=2,
    num_encoder_layers=2,
    dim_feedforward=128,
    dropout=0.0,
    maxlen=32,
).to(device)


cleaned_train = load_or_clean_data("train[:1%]")
cleaned_val = load_or_clean_data("validation[:10%]")

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data" / "tokenizer"

custom_tokenizer = CustomTokenizer(
    vocab_size=50000, corpus_file=str(data_dir / "byte-level-bpe_wmt17.tokenizer.json")
)
tokenizer = custom_tokenizer.load_gpt2_tokenizer()  # Load pre-trained tokenizer

train_dataset = TranslationDataset(cleaned_train, tokenizer=tokenizer)
validation_dataset = TranslationDataset(cleaned_val, tokenizer=tokenizer)
# test_dataset = TranslationDataset(dataset["test"], tokenizer=tokenizer)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

params_with_decay = []
params_without_decay = []

no_decay = ["bias", "layer_norm"]
for name, param in model.named_parameters():
    if any(nd in name for nd in no_decay):
        params_without_decay.append(param)
    else:
        params_with_decay.append(param)

adamw_optimizer = optim.AdamW(
    [
        {"params": params_with_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ],
    lr=1e-4,
    weight_decay=1e-2,
)

criterion = nn.CrossEntropyLoss()

lr_scheduler = LR_Scheduler(adamw_optimizer, d_model=32, warmup_steps=1000)


def train_and_validate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    num_epochs=1,
    vocab_size=50000,
):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            X_batch = X_batch.to(device)
            #  print(f"X_batch.shape: {X_batch.shape}")
            #  print(f"X_batch: {X_batch}")
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch, y_batch)

            # print(f"preds.shape: {preds.shape}")
            # print(f"Predictions: {preds.argmax(-1)}")

            translation = tokenizer.decode(
                preds.argmax(-1)[0], skip_special_tokens=True
            )
            # print(f"Translation: {translation}")

            # preds.shape = [batch_size, seq_len, vocab_size]
            # y_batch.shape = [batch_size, seq_len]

            loss = criterion(preds.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(
                val_dataloader, desc=f"Validating Epoch {epoch}"
            ):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch, y_batch)
                loss = criterion(preds.view(-1, vocab_size), y_batch.view(-1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(validation_dataloader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

    return train_losses, val_losses


train_losses, val_losses = train_and_validate(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=validation_dataloader,
    optimizer=adamw_optimizer,
    criterion=criterion,
    vocab_size=50000,
)
