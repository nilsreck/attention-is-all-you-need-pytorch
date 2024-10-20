from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(
    vocab_size=295,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train([
    "/home/reck/personal/transformer_project/transformer_project/run/corpus.txt",
], trainer=trainer)

# And Save it
tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)

test = "Machine learning is a subset of artificial intelligence."
encoded = tokenizer.encode(test)
print(encoded.tokens)
