import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import time
import json
import numpy as np
import psutil


from transformer_project.modelling.transformer import Transformer
from transformer_project.data.translation_dataset import TranslationDataset
from transformer_project.modelling.lr_scheduler import LR_Scheduler

# from transformer_project.preprocessing.clean_data import load_or_clean_data
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer
from transformer_project.run.inference import translate
from transformer_project.run.bleu import _compute_bleu

# BATCH_SIZE = 1500 https://arxiv.org/pdf/1804.00247
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NUM_HEADS = 8
D_MODEL = 64
DIM_FEED_FORWARD = 4 * D_MODEL
DROPOUT = 0.0
MAX_LEN = 64
VOCAB_SIZE = 50000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 10
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=NUM_HEADS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feed_forward=DIM_FEED_FORWARD,
    dropout=DROPOUT,
    maxlen=MAX_LEN,
).to(DEVICE)


# cleaned_train = load_or_clean_data("train[:1%]")
# cleaned_val = load_or_clean_data("validation[:160]")

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "transformer_project" / "data"

tokenizer = CustomTokenizer(
    vocab_size=VOCAB_SIZE,
    corpus_file=str(data_dir / "tokenizer" / "byte-level-bpe_wmt17.tokenizer.json"),
).load_gpt2_tokenizer()

train_data = torch.load(data_dir / "train_dataset.pt")
val_data = torch.load(data_dir / "val_dataset.pt")

train_dataset = TranslationDataset(train_data, tokenizer=tokenizer)
val_dataset = TranslationDataset(val_data, tokenizer=tokenizer)


train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)

params_with_decay = []
params_without_decay = []

no_decay = ["bias", "layer_norm"]
for name, param in model.named_parameters():
    if any(nd in name for nd in no_decay):
        params_without_decay.append(param)
    else:
        params_with_decay.append(param)

optimizer = optim.AdamW(
    [
        {"params": params_with_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

lr_scheduler = LR_Scheduler(optimizer, d_model=D_MODEL, warmup_steps=WARMUP_STEPS)

scaler = torch.cuda.amp.GradScaler()


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_memory_usage(
    device, epoch=None, log_file=f"memory_usage_{get_timestamp()}.log"
):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_str = f"Epoch {epoch}: " if epoch is not None else "Start: "

        if device.type == "cuda":
            gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)
            f.write(
                f"[{timestamp}] {epoch_str}GPU Memory Allocated: {gpu_memory:.2f} MB\n"
            )
        else:
            cpu_memory = psutil.virtual_memory().used / (1024**3)
            f.write(f"[{timestamp}] CPU Memory Used: {cpu_memory:.2f} GB\n")


def train_and_validate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    lr_scheduler,
    tokenizer,
    device,
    num_epochs=NUM_EPOCHS,
    vocab_size=50000,
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    bleu_scores = []
    timing_metrics = {
        "epoch_times": [],
        "forward_times": [],
        "backward_times": [],
        "device": str(device),
    }

    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch + 1} / {num_epochs} ===")
        model.train()
        train_loss_total = 0.0
        epoch_start = time.time()
        forward_times = []
        backward_times = []

        for _, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
        ):
            src_input, tgt_input, tgt_output = (
                batch["source"],
                batch["target_input"],
                batch["target_output"],
            )

            # source_text = tokenizer.decode(src_input[0], skip_special_tokens=True)
            # print(f"Source text: {source_text}")

            src_input, tgt_input, tgt_output = (
                src_input.to(DEVICE),
                tgt_input.to(DEVICE),
                tgt_output.to(DEVICE),
            )

            enc_att_mask = (src_input != tokenizer.pad_token_id).int().to(device)
            dec_att_mask = (tgt_input != tokenizer.pad_token_id).int().to(device)

            optimizer.zero_grad()

            forward_start = time.time()
            with torch.cuda.amp.autocast():
                preds = model(
                    src_input,
                    tgt_input,
                    encoder_attention_mask=enc_att_mask,
                    decoder_attention_mask=dec_att_mask,
                )
                loss = criterion(preds.view(-1, vocab_size), tgt_output.view(-1))

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_times.append(time.time() - forward_start)

            backward_start = time.time()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            backward_times.append(time.time() - backward_start)

            train_loss_total += loss.item()

        timing_metrics["epoch_times"].append(time.time() - epoch_start)
        timing_metrics["forward_times"].append(sum(forward_times) / len(forward_times))
        timing_metrics["backward_times"].append(
            sum(backward_times) / len(backward_times)
        )

        print(f"Epoch {epoch} timing:")
        print(f"Total time: {timing_metrics['epoch_times'][-1]:.2f}s")
        print(f"Avg forward pass: {timing_metrics['forward_times'][-1]:.4f}s")
        print(f"Avg backward pass: {timing_metrics['backward_times'][-1]:.4f}s")
        log_memory_usage(device, epoch)

        avg_train_loss = train_loss_total / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_total = 0.0

        bleu_predictions = []
        bleu_references = []

        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_dataloader)):

                src_input, tgt_input, tgt_output = (
                    batch["source"].to(device),
                    batch["target_input"].to(device),
                    batch["target_output"].to(device),
                )

                enc_att_mask = (src_input != tokenizer.pad_token_id).int().to(device)

                decoded_preds, logits = translate(
                    model, src_seq=src_input, tokenizer=tokenizer, device=device
                )

                loss = criterion(logits.view(-1, vocab_size), tgt_output.view(-1))
                val_loss_total += loss.item()

                bleu_predictions.extend(decoded_preds)
                bleu_references.extend(
                    [
                        tokenizer.decode(ref, skip_special_tokens=True)
                        for ref in tgt_output
                    ]
                )

        bleu_score = _compute_bleu(
            predictions=bleu_predictions, references=bleu_references
        )
        print(f"BLEU Score: {bleu_score}")
        bleu_scores.append(bleu_score["bleu"])
        print(f"BLEU Scores: {bleu_scores}")

        avg_val_loss = val_loss_total / len(val_dataloader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch} => "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    with open(f"timing_metrics_{device}_{get_timestamp()}.json", "w") as f:
        json.dump(timing_metrics, f)

    return train_losses, val_losses, bleu_scores, timing_metrics


train_losses, val_losses, bleu_scores, timing_metrics = train_and_validate(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    lr_scheduler=lr_scheduler,
    tokenizer=tokenizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    vocab_size=50000,
)


epoch_times = timing_metrics["epoch_times"]
cumulative_hours = np.cumsum(epoch_times) / 3600

print(f"Cumulative training time: {cumulative_hours}")


def plot_lr_schedule(lr_scheduler, num_epochs):
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    lrs = lr_scheduler.get_lr_schedule(total_steps)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_steps + 1), lrs)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.savefig(f"lr_schedule_{get_timestamp()}.png")
    plt.close()


def plot_bleu_score(bleu_scores, cumulative_hours):
    plt.plot(cumulative_hours, bleu_scores, marker="o")
    plt.xlabel("Training Time (hours)")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score vs Training Time")
    plt.grid(True)
    plt.savefig(f"bleu_score_val_{get_timestamp()}.png")


plot_bleu_score(bleu_scores, cumulative_hours)
plot_lr_schedule(lr_scheduler, NUM_EPOCHS)
