from datasets import load_dataset
import json
import os
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformer_project.preprocessing.clean_data import clean_data
from pathlib import Path

PATH = Path(
    "/home/reck/personal/transformer_project/transformer_project/data/tokenizer"
)

VOCAB_SIZE = 50000
MIN_FREQ = 2


class CustomTokenizer:
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        corpus_file=PATH / "wmt17_cleaned_corpus.txt",
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.corpus_file = corpus_file
        self.special_tokens = {
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
        }

    def load_and_clean_data(self):
        corpus_path = Path(self.corpus_file)
        cleaned_data_path = corpus_path.parent / "cleaned_dataset.json"
        os.makedirs(corpus_path.parent, exist_ok=True)

        if cleaned_data_path.exists():
            print(f"Loading cached cleaned data from {cleaned_data_path}")
            with open(cleaned_data_path, "r") as f:
                return json.load(f)

        print("Cleaning dataset...")
        train_dataset = load_dataset("wmt17", "de-en", split="train")
        cleaned_train_data = clean_data(train_dataset)

        with open(cleaned_data_path, "w") as f:
            json.dump(cleaned_train_data, f)

        return cleaned_train_data

    def train_tokenizer(self):
        if not os.path.exists(self.corpus_file):
            self.load_and_clean_data()

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=list(self.special_tokens.values()),
        )

        tokenizer.train(files=[str(self.corpus_file)], trainer=trainer)

        vocab_path = PATH / "vocab.json"
        merges_path = PATH / "merges.txt"

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=4)

        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in tokenizer.model.__getstate__()["merges"]:
                f.write(" ".join(merge) + "\n")

    def load_gpt2_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained(str(PATH))
        tokenizer.add_special_tokens(self.special_tokens)
        return tokenizer

    def build_tokenizer(self):
        if not (PATH / "vocab.json").exists() or not (PATH / "merges.txt").exists():
            self.train_tokenizer()
        return self.load_gpt2_tokenizer()
