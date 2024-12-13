import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm
from datasets import load_dataset

from transformer_project.modelling.transformer import Transformer
from transformer_project.modelling import huggingface_bpe_tokenizer
from transformer_project.data.translation_dataset import TranslationDataset
from transformer_project.modelling.lr_scheduler import LR_Scheduler
from transformer_project.preprocessing.clean_data import clean_data
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

# Initialize the data loader
dataset = load_dataset("wmt17", "de-en", split="train[:5%]")
cleaned_dataset = clean_data(dataset)
custom_tokenizer = CustomTokenizer()
tokenizer = custom_tokenizer.build_tokenizer()

train_dataset = TranslationDataset(cleaned_dataset, tokenizer=tokenizer)
# validation_dataset = TranslationDataset(dataset["validation"], tokenizer=tokenizer)
# test_dataset = TranslationDataset(dataset["test"], tokenizer=tokenizer)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the AdamW optimizer
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

# Initialize the loss function
criterion = nn.CrossEntropyLoss()

# Initialize the learning rate scheduler
lr_scheduler = LR_Scheduler(adamw_optimizer, d_model=32, warmup_steps=1000)


def train(model, dataloader, optimizer, criterion, num_epochs=5):
    model.train()

    train_losses = []

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0.0
        for X_batch, y_batch in tqdm(dataloader, desc="Batches", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            print(f"X_batch shape: {X_batch.shape}")
            print(f"y_batch shape: {y_batch.shape}")
            print(f"y_batch max index: {y_batch.max()}")

            optimizer.zero_grad()
            preds = model(X_batch, y_batch)
            print(f"preds shape: {preds.shape}")

            loss = criterion(preds, y_batch)
            epoch_loss += loss.item() * X_batch.size(0)

            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader.dataset)
        train_losses.append(epoch_loss)

        lr_scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    return train_losses


train(
    model=model,
    dataloader=train_dataloader,
    optimizer=adamw_optimizer,
    criterion=criterion,
)
