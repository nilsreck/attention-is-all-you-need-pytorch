import re
from typing import List, Dict
from datasets import DatasetDict
from datasets import load_dataset
import json
from pathlib import Path


WHITELIST = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"


def load_or_clean_data(split="train"):
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cleaned_data_path = data_dir / f"cleaned_dataset_{split}.json"

    if cleaned_data_path.exists():
        print(f"Loading cached {split} data from {cleaned_data_path}")
        with open(cleaned_data_path, "r") as f:
            return json.load(f)

    print(f"Creating new cleaned dataset for {split}...")
    dataset_raw = load_dataset("wmt17", "de-en", split=split)
    cleaned_data = clean_data(dataset_raw)

    cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned data to {cleaned_data_path}")
    with open(cleaned_data_path, "w") as f:
        json.dump(cleaned_data, f)

    return cleaned_data


def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r"http\S+|www.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only whitelisted characters
    text = "".join([char for char in text if char in WHITELIST])

    # Convert to lowercase
    text = text.lower()

    return text


def filter_sentence_length(
    sentence: str, min_length: int = 5, max_length: int = 64
) -> bool:
    return min_length <= len(sentence.split()) <= max_length


def filter_ratio(source_seq: str, target_seq: str, max_ratio: float = 1.5) -> bool:
    return source_seq / target_seq <= max_ratio or target_seq / source_seq


def clean_data(
    dataset: DatasetDict,
    min_length: int = 5,
    max_length: int = 64,
    max_ratio: float = 1.5,
) -> DatasetDict:
    cleaned_data: List[Dict[str, str]] = []

    for example in dataset:
        source: str = example["translation"]["de"]
        target: str = example["translation"]["en"]

        source = clean_text(source)
        target = clean_text(target)

        if not filter_sentence_length(source, min_length, max_length):
            continue
        if not filter_sentence_length(target, min_length, max_length):
            continue

        source_len, target_len = len(source.split()), len(target.split())
        if source_len / target_len > max_ratio or target_len / source_len > max_ratio:
            continue

        cleaned_data.append({"de": source, "en": target})

    return cleaned_data
