import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
from transformer_project.modelling.transformer import Transformer
from transformer_project.data.translation_dataset import TranslationDataset
from transformer_project.preprocessing.clean_data import load_or_clean_data
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer

BATCH_SIZE = 32
D_MODEL = 64
DIM_FEED_FORWARD = 4 * D_MODEL
VOCAB_SIZE = 50000
NUM_HEADS = 8
NUM_DEC_LAYERS = 4
NUM_ENC_LAYERS = 4
DROPOUT = 0.0
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, dim_feedforward: int, device: str, d_model: int):
    model = Transformer(
        d_model=d_model,
        vocab_size=VOCAB_SIZE,
        n_heads=NUM_HEADS,
        num_decoder_layers=NUM_DEC_LAYERS,
        num_encoder_layers=NUM_ENC_LAYERS,
        dim_feed_forward=DIM_FEED_FORWARD,
        dropout=DROPOUT,
        maxlen=MAX_LEN,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path))
    #     print("Model's state_dict:")
    #     for param_tensor in model.state_dict():
    #         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return model.to(device)


def translate(model, src_seq, tokenizer, device="cuda", max_len=64):
    model.eval()
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        enc_mask = (src_seq != pad_token_id).int()

        init_dec_input = torch.tensor(
            [[bos_token_id] + [pad_token_id] * (max_len - 1)], device=device
        ).repeat(
            BATCH_SIZE, 1
        )  # [batch_size, seq_length]
        print(f"dec input: {init_dec_input}")
        dec_att_mask = (init_dec_input != tokenizer.pad_token_id).int().to(DEVICE)

        eos_generated = torch.zeros(src_seq.size(0), dtype=torch.bool, device=device)

        for i in range(1, max_len - 1):
            dec_att_mask = (init_dec_input != tokenizer.pad_token_id).int().to(DEVICE)
            print(f"src seq dims: {src_seq.shape}")
            output = model(
                src_seq, init_dec_input, enc_mask, dec_att_mask
            )  # [batch_size, seq_length, vocab_size]
            print(f"output.shape = {output.shape}")
            next_tokens = torch.argmax(output[:, i, :], dim=-1)  # [batch_size]
            print(f"next_tokens: {next_tokens}")
            print(f"next_tokens.shape: {next_tokens.shape}")

            init_dec_input[:, i] = torch.where(eos_generated, pad_token_id, next_tokens)
            print(f"next_token.shape: {next_tokens.shape}")

            eos_generated |= next_tokens == tokenizer.eos_token_id
            print(f"init_dec_input.shape = {init_dec_input}")

            if eos_generated.all():
                break

    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in init_dec_input]


if __name__ == "__main__":
    model = load_model("best_model.pth", DIM_FEED_FORWARD, DEVICE, D_MODEL)

    tokenizer = CustomTokenizer(
        vocab_size=VOCAB_SIZE, corpus_file=str("byte-level-bpe_wmt17.tokenizer.json")
    ).load_gpt2_tokenizer()

    cleaned_test_data = load_or_clean_data("test[:10%]")
    test_dataset = TranslationDataset(cleaned_test_data, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    iterator = tqdm(test_dataloader, desc="Test")

    translations = []
    references = []

    for batch in iterator:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        src_seq = batch["source"]
        tgt_output = batch["target_output"]

        translated_batch = translate(model, src_seq, tokenizer, DEVICE)
        translations.extend(translated_batch)

        references.extend(
            [tokenizer.decode(ref, skip_special_tokens=True) for ref in tgt_output]
        )

    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(
        predictions=translations, references=[[ref] for ref in references]
    )
    print(f"BLEU Score: {bleu_score['bleu']}")

    plt.plot(
        range(len(translations)), [bleu_score["bleu"]] * len(translations), marker="o"
    )
    plt.xlabel("Translation Steps")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score vs Translation Steps")
    plt.grid(True)
    plt.savefig("bleu_score_test.png")
