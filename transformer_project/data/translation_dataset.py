import torch
from torch.utils.data import Dataset

MAX_LEN = 64


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        transform=None,
        max_len=MAX_LEN,
    ):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

        self.valid_indices = []
        for idx in range(len(dataset)):
            source_ids = tokenizer.encode(
                dataset[idx]["de"], add_special_tokens=False, truncation=False
            )
            target_ids = tokenizer.encode(
                dataset[idx]["en"], add_special_tokens=False, truncation=False
            )
            if len(source_ids) <= max_len and len(target_ids) <= max_len:
                self.valid_indices.append(idx)

        self.dataset = [dataset[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_text = self.dataset[idx]["de"]
        target_text = self.dataset[idx]["en"]

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        encoder_input = torch.tensor(
            self.tokenizer.encode(
                source_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
        )

        encoded_target = torch.tensor(
            self.tokenizer.encode(
                target_text,
                max_length=self.max_len - 1,
                padding="max_length",
                truncation=True,
            )
        )

        encoded_target_full = torch.tensor(
            self.tokenizer.encode(
                target_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
        )

        decoder_input = torch.cat([torch.tensor([bos_token_id]), encoded_target])

        last_token_idx = (encoded_target_full != self.tokenizer.pad_token_id).nonzero()[
            -1
        ]
        decoder_target = encoded_target_full.clone()
        if last_token_idx <= MAX_LEN - 2:
            decoder_target[last_token_idx + 1] = eos_token_id

        if self.transform:
            encoder_input = self.transform(encoder_input)
            encoder_target_output = self.transform(encoder_target_output)

        return {
            "source": encoder_input,
            "target_input": decoder_input,
            "target_output": decoder_target,
        }
