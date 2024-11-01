from datasets import load_dataset, DatasetDict
import json
from transformer_project.preprocessing.clean_data import clean_data
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def load_and_clean_data():
    train_dataset = load_dataset("wmt17", "de-en", split="train")

    cleaned_train_data = clean_data(train_dataset)

    corpus = []
    for example in cleaned_train_data:
        german_sentence = example["de"]
        english_sentence = example["en"]
        corpus.append(german_sentence)
        corpus.append(english_sentence)

    with open("wmt17_corpus.txt", "w", encoding="utf-8") as f:
        for sentence in corpus:
            f.write(sentence + "\n")


def train_tokenizer():
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    special_tokens = ["[PAD]", "[BOS]", "[EOS]"]
    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=50000,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=special_tokens,
    )

    # Then train it!
    tokenizer.train(
        [
            "/home/reck/personal/transformer_project/transformer_project/data/wmt17_corpus.txt"
        ],
        trainer=trainer,
    )

    # And save it
    tokenizer.save("byte-level-bpe_wmt17.tokenizer.json", pretty=True)


def convert_to_GPT2_format():
    # Conversion to GPT2Tokenizer
    with open(
        "/home/reck/personal/transformer_project/transformer_project/run/byte-level-bpe_wmt17.tokenizer.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # Extract the vocabulary dictionary
    vocab_dict = data["model"]["vocab"]
    merges_dict = data["model"]["merges"]

    with open(
        "/home/reck/personal/transformer_project/transformer_project/data/vocab.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

    # with open(
    # "/home/reck/personal/transformer_project/transformer_project/data/merges.json",
    # "w",
    # encoding="utf-8",
    # ) as f:
    # json.dump(merges_dict, f, ensure_ascii=False, indent=4)

    # Save merges.txt in the expected format
    with open(
        "/home/reck/personal/transformer_project/transformer_project/data/tokenizer_files/merges.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for merge in merges_dict:
            f.write(" ".join(merge) + "\n")


def load_gpt2_tokenizer():

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
        "/home/reck/personal/transformer_project/transformer_project/data/tokenizer_files"
    )

    return gpt2_tokenizer


gpt2_tokenizer = load_gpt2_tokenizer()

