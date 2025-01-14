import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
import evaluate
import matplotlib.pyplot as plt


from transformer_project.modelling.transformer import Transformer
from transformer_project.data.translation_dataset import TranslationDataset
from transformer_project.modelling.lr_scheduler import LR_Scheduler
from transformer_project.preprocessing.clean_data import load_or_clean_data
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer

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
WARMUP_STEPS = 1
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


cleaned_train = load_or_clean_data("train[:10%]")
cleaned_val = load_or_clean_data("validation[:50%]")

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data" / "tokenizer"

tokenizer = CustomTokenizer(
    vocab_size=VOCAB_SIZE,
    corpus_file=str(data_dir / "byte-level-bpe_wmt17.tokenizer.json"),
).load_gpt2_tokenizer()

train_dataset = TranslationDataset(cleaned_train, tokenizer=tokenizer)
val_dataset = TranslationDataset(cleaned_val, tokenizer=tokenizer)
# test_dataset = TranslationDataset(dataset["test"], tokenizer=tokenizer)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch + 1} / {num_epochs} ===")
        model.train()
        train_loss_total = 0.0

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

            enc_att_mask = (src_input != tokenizer.pad_token_id).int().to(DEVICE)
            dec_att_mask = (tgt_input != tokenizer.pad_token_id).int().to(DEVICE)

            optimizer.zero_grad()
            preds = model(
                src_input,
                tgt_input,
                encoder_attention_mask=enc_att_mask,
                decoder_attention_mask=dec_att_mask,
            )
            # preds: [batch_size, seq_len, vocab_size]

            loss = criterion(preds.view(-1, vocab_size), tgt_output.view(-1))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_total = 0.0
        bleu = evaluate.load("bleu")

        bleu_predictions = []
        bleu_references = []

        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_dataloader)):

                src_input, tgt_input, tgt_output = (
                    batch["source"],
                    batch["target_input"],
                    batch["target_output"],
                )

                src_input, tgt_input, tgt_output = (
                    src_input.to(DEVICE),
                    tgt_input.to(DEVICE),
                    tgt_output.to(DEVICE),
                )

                enc_att_mask = (src_input != tokenizer.pad_token_id).int().to(DEVICE)
                dec_att_mask = (tgt_input != tokenizer.pad_token_id).int().to(DEVICE)

                # TODO: remove teacher forcing
                preds = model(
                    src_input,
                    tgt_input,
                    encoder_attention_mask=enc_att_mask,
                    decoder_attention_mask=dec_att_mask,
                )
                # preds: [batch_size, seq_len, vocab_size]

                loss = criterion(preds.view(-1, vocab_size), tgt_output.view(-1))

                val_loss_total += loss.item()

                batch_predictions = [
                    tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in preds.argmax(dim=-1)
                ]
                batch_references = [
                    [tokenizer.decode(ref, skip_special_tokens=True)]
                    for ref in tgt_output
                ]

                bleu_predictions.extend(batch_predictions)
                bleu_references.extend(batch_references)

        bleu_score = bleu.compute(
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

    return train_losses, val_losses, bleu_scores


train_losses, val_losses, bleu_scores = train_and_validate(
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

training_steps = list(range(NUM_EPOCHS))
plt.plot(training_steps, bleu_scores, marker="o")
plt.xlabel("Training Steps")
plt.ylabel("BLEU Score")
plt.title("BLEU Score vs Training Steps")
plt.grid(True)
plt.savefig("bleu_score_val.png")
