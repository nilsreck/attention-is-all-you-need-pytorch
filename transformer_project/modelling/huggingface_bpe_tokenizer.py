from datasets import load_dataset
import json
import os
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformer_project.preprocessing.clean_data import clean_data
from pathlib import Path


class CustomTokenizer:
    def __init__(
        self,
        vocab_size=50000,
        min_frequency=2,
        corpus_file="/home/reck/personal/transformer_project/transformer_project/data/tokenizer/wmt17_cleaned_corpus.txt",
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

        # Save cleaned data
        with open(cleaned_data_path, "w") as f:
            json.dump(cleaned_train_data, f)

        with open(self.corpus_file, "w") as f:
            for item in cleaned_train_data:
                f.write(item["de"] + "\n")
                f.write(item["en"] + "\n")

        return cleaned_train_data

    def train_tokenizer(self):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=list(self.special_tokens.values()),
            show_progress=True,
        )

        tokenizer.train([self.corpus_file], trainer=trainer)
        tokenizer.save(
            "/home/reck/personal/transformer_project/transformer_project/data/tokenizer/byte-level-bpe_wmt17.tokenizer.json",
            pretty=True,
        )

    def convert_to_gpt2_format(self):
        with open(
            "/home/reck/personal/transformer_project/transformer_project/data/tokenizer/byte-level-bpe_wmt17.tokenizer.json",
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        vocab_dict = data["model"]["vocab"]
        merges_list = data["model"]["merges"]

        with open(
            "/home/reck/personal/transformer_project/transformer_project/data/tokenizer/vocab.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

        with open(
            "/home/reck/personal/transformer_project/transformer_project/data/tokenizer/merges.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for merge in merges_list:
                f.write(" ".join(merge) + "\n")

    def load_gpt2_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained(
            "/home/reck/personal/transformer_project/transformer_project/data/tokenizer",
        )
        tokenizer.add_special_tokens(self.special_tokens)
        return tokenizer

    def build_tokenizer(self):
        if not os.path.exists(self.corpus_file):
            self.load_and_clean_data()
        if not os.path.exists(
            "/home/reck/personal/transformer_project/transformer_project/data/tokenizer/vocab.json"
        ) or not os.path.exists(
            "/home/reck/personal/transformer_project/transformer_project/data/tokenizer/merges.txt"
        ):
            self.train_tokenizer()
            self.convert_to_gpt2_format()
        return self.load_gpt2_tokenizer()
