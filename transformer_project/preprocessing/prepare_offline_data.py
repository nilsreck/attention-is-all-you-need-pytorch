import torch
from pathlib import Path
from datasets import load_dataset
from transformer_project.preprocessing.clean_data import clean_data


def prepare_and_save_data(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_dataset("wmt17", "de-en", split="train[:5%]")
    val_data = load_dataset("wmt17", "de-en", split="validation[:10%]")
    test_data = load_dataset("wmt17", "de-en", split="test")

    train_cleaned = clean_data(train_data)
    val_cleaned = clean_data(val_data)
    test_cleaned = clean_data(test_data)

    torch.save(train_cleaned, save_dir / "train_dataset.pt")
    torch.save(val_cleaned, save_dir / "val_dataset.pt")
    torch.save(test_cleaned, save_dir / "test_dataset.pt")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    prepare_and_save_data(data_dir)
