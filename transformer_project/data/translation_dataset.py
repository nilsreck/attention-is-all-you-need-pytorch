import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        source_lang="de",
        target_lang="en",
        transform=None,
        max_len=32,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_text = self.dataset[idx][self.source_lang]
        target_text = self.dataset[idx][self.target_lang]
        source_tokens = torch.tensor(
            self.tokenizer.encode(
                source_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
        )
        target_tokens = torch.tensor(
            self.tokenizer.encode(
                target_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
        )

        if self.transform:
            source_tokens = self.transform(source_tokens)
            target_tokens = self.transform(target_tokens)

        return source_tokens, target_tokens
