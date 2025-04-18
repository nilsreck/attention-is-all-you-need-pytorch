import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
import itertools
from transformer_project.modelling.transformer import Transformer
from transformer_project.data.translation_dataset import TranslationDataset
from transformer_project.preprocessing.clean_data import load_or_clean_data
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer
from transformer_project.run.bleu import _compute_bleu

BATCH_SIZE = 32
D_MODEL = 512
DIM_FEED_FORWARD = 4 * D_MODEL
VOCAB_SIZE = 50000
NUM_HEADS = 8
NUM_DEC_LAYERS = 6
NUM_ENC_LAYERS = 6
DROPOUT = 0.1
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

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
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
        batch_size = src_seq.size(0)

        init_dec_input = torch.tensor(
            [[bos_token_id] + [pad_token_id] * (max_len - 1)], device=device
        ).repeat(batch_size, 1)

        eos_generated = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(1, max_len - 1):
            output = model(src_seq, init_dec_input, enc_mask)
            next_tokens = torch.argmax(output[:, i - 1, :], dim=-1)

            init_dec_input[:, i] = torch.where(eos_generated, pad_token_id, next_tokens)
            eos_generated |= next_tokens == tokenizer.eos_token_id

            if eos_generated.all():
                break

        decoded_seqs = [
            tokenizer.decode(seq, skip_special_tokens=True) for seq in init_dec_input
        ]
        return decoded_seqs, output


def print_translations(translations, references, sources):
    print("\n=== Translation Results ===")
    for i, (src, ref, trans) in enumerate(zip(sources, references, translations), 1):
        print(f"\nExample {i}:")
        print(f"Source:      {src}")
        print(f"Reference:   {ref}")
        print(f"Translation: {trans}")
        print("-" * 80)


if __name__ == "__main__":
    model = load_model("best_model.pth", DIM_FEED_FORWARD, DEVICE, D_MODEL)

    tokenizer = CustomTokenizer(
        vocab_size=VOCAB_SIZE, corpus_file=str("byte-level-bpe_wmt17.tokenizer.json")
    ).load_gpt2_tokenizer()

    cleaned_test_data = load_or_clean_data("test")
    test_dataset = TranslationDataset(cleaned_test_data, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    iterator = tqdm(test_dataloader, desc="Test")

    translations = []
    references = []
    sources = []

    for batch in itertools.islice(iterator, len(iterator) - 1):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        src_seq = batch["source"]
        tgt_output = batch["target_output"]

        translated_batch, _ = translate(model, src_seq, tokenizer, DEVICE)
        translations.extend(translated_batch)
        references.extend(
            [tokenizer.decode(ref, skip_special_tokens=True) for ref in tgt_output]
        )
        sources.extend(
            [tokenizer.decode(src, skip_special_tokens=True) for src in src_seq]
        )

    for i, (source, reference, translation) in enumerate(
        zip(sources, references, translations)
    ):
        print(f"\nExample {i + 1}:")
        print(f"Source:      {source}")
        print(f"Reference:   {reference}")
        print(f"Translation: {translation}")
        print("-" * 80)

    bleu_score = _compute_bleu(
        predictions=translations, references=[[ref] for ref in references]
    )
    print(f"BLEU Score: {bleu_score}")
