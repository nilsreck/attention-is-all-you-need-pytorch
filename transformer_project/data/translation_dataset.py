import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        transform=None,
        max_len=64,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_text = self.dataset[idx]["de"]
        target_text = self.dataset[idx]["en"]

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        encoded_source = torch.tensor(
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

        encoder_target_input = torch.cat([torch.tensor([bos_token_id]), encoded_target])
        encoder_target_output = torch.cat(
            [encoded_target, torch.tensor([eos_token_id])]
        )

        if self.transform:
            encoded_source = self.transform(encoded_source)
            encoder_target_output = self.transform(encoder_target_output)

        return {
            "source": encoded_source,
            "target_input": encoder_target_input,
            "target_output": encoder_target_output,
        }
