from pathlib import Path
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer


def train_wmt17_tokenizer():
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "transformer_project" / "data" / "tokenizer"
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = data_dir / "wmt_17_cleaned_corpus.txt"

    tokenizer = CustomTokenizer(
        vocab_size=50000,
        min_frequency=2,
        corpus_file=str(corpus_file),
    )

    if corpus_file.exists():
        print(f"Corpus file already exists at: {corpus_file}")
    else:
        print("Loading and cleaning WMT17 dataset...")
        tokenizer.load_and_clean_data()

    print("Training tokenizer...")
    tokenizer.train_tokenizer()

    print("Converting to GPT2 format...")
    tokenizer.convert_to_gpt2_format()

    print("Done! Tokenizer files saved in:", data_dir)


if __name__ == "__main__":
    train_wmt17_tokenizer()
