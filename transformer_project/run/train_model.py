import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, Subset
import torch.utils.data
from tqdm import tqdm
from datasets import load_dataset

from transformer_project.modelling.transformer import Transformer
from transformer_project.modelling import huggingface_bpe_tokenizer
from transformer_project.data.translation_dataset import TranslationDataset
from transformer_project.modelling.lr_scheduler import LR_Scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    vocab_size=50000,
    d_model=32,
    n_heads=8,
    num_decoder_layers=2,
    num_encoder_layers=2,
    dim_feedforward=128,
    dropout=0.0,
    maxlen=8,
).to(device)

# Initialize the data loader
dataset = load_dataset("wmt17", "de-en", split="train[:5%]")
custom_tokenizer = huggingface_bpe_tokenizer.CustomTokenizer()
tokenizer = custom_tokenizer.build_tokenizer()

train_dataset = TranslationDataset(dataset, tokenizer=tokenizer)
# validation_dataset = TranslationDataset(dataset["validation"], tokenizer=tokenizer)
# test_dataset = TranslationDataset(dataset["test"], tokenizer=tokenizer)


def collate_fn(batch):
    source_batch, target_batch = zip(*batch)
    source_batch = [torch.tensor(tokens) for tokens in source_batch]
    target_batch = [torch.tensor(tokens) for tokens in target_batch]
    source_padded = pad_sequence(source_batch, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_batch, batch_first=True, padding_value=0)
    return source_padded, target_padded


train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
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
criterion = nn.CrossEntropyLoss

# Initialize the learning rate scheduler
lr_scheduler = LR_Scheduler(adamw_optimizer, d_model=32, warmup_steps=1000)


def train(model, dataloader, optimizer, criterion, num_epochs=5):
    model.train()

    losses = []

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0.0
        for X_batch, y_batch in tqdm(dataloader, desc="Batches", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            preds = model(X_batch)

            loss = criterion(preds, y_batch)
            epoch_loss += loss.item() * X_batch.size(0)

            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    return losses


train(
    model=model,
    dataloader=train_dataloader,
    optimizer=adamw_optimizer,
    criterion=criterion,
)
