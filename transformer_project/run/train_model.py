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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5

d_model = 64
dim_feed_forward = 4 * d_model

model = Transformer(
    vocab_size=50000,
    d_model=d_model,
    n_heads=8,
    num_decoder_layers=4,
    num_encoder_layers=4,
    dim_feed_forward=dim_feed_forward,
    dropout=0.0,
    maxlen=64,
).to(device)


cleaned_train = load_or_clean_data("train[:1%]")
cleaned_val = load_or_clean_data("validation")

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

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

lr_scheduler = LR_Scheduler(adamw_optimizer, d_model=d_model, warmup_steps=200)


def train_and_validate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    lr_scheduler,
    tokenizer,
    device,
    num_epochs=5,
    vocab_size=50000,
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch} / {num_epochs} ===")
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

            src_input, tgt_input, tgt_output = (
                src_input.to(device),
                tgt_input.to(device),
                tgt_output.to(device),
            )

            enc_att_mask = (src_input != tokenizer.pad_token_id).int().to(device)
            dec_att_mask = (tgt_input != tokenizer.pad_token_id).int().to(device)

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
        bleu_scores = []
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
                    src_input.to(device),
                    tgt_input.to(device),
                    tgt_output.to(device),
                )

                enc_att_mask = (src_input != tokenizer.pad_token_id).int().to(device)
                dec_att_mask = (tgt_input != tokenizer.pad_token_id).int().to(device)

                optimizer.zero_grad()
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
        bleu_scores.append(bleu_score)

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
    val_dataloader=validation_dataloader,
    optimizer=adamw_optimizer,
    criterion=criterion,
    lr_scheduler=lr_scheduler,
    tokenizer=tokenizer,
    device=device,
    num_epochs=5,
    vocab_size=50000,
)

training_steps = [range(NUM_EPOCHS)]
plt.plot(training_steps, bleu_scores, marker="o")
plt.xlabel("Training Steps")
plt.ylabel("BLEU Score")
plt.title("BLEU Score vs Training Steps")
plt.grid(True)
plt.show()
