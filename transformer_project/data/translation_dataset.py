from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self, dataset, tokenizer, source_lang="de", target_lang="en", transform=None
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.dataset[idx]["translation"][self.source_lang]
        target_text = self.dataset[idx]["translation"][self.target_lang]
        source_tokens = self.tokenizer.encode(source_text)
        target_tokens = self.tokenizer.encode(target_text)

        if self.transform:
            source_tokens = self.transform(source_tokens)
            target_tokens = self.transform(target_tokens)

        return source_tokens, target_tokens
