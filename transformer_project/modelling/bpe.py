from collections import Counter
import string

corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning.",
]

text = " ".join(corpus)
text = text.translate(str.maketrans("", "", string.punctuation))
text = text.lower()

words = text.split()

word_counter = Counter(words)

vocabulary = []
for s in text:
    if s not in vocabulary and s != " ":
        vocabulary.extend(s)

splits = {word: [c for c in word] for word in word_counter.keys()}


def compute_pair_freqs(splits):
    pair_freqs = Counter()
    for word, freq in word_counter.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


pair_freqs = compute_pair_freqs(splits)

rules = {}


def create_rule(pair_freqs, rules, vocabulary):
    most_common_pair = pair_freqs.most_common(1)[0]
    concatenated = "".join(most_common_pair[0])
    rules[most_common_pair[0]] = concatenated
    vocabulary.append(concatenated)
    pair_freqs.pop(most_common_pair[0])


def replace_in_splits(splits, rules):
    for _, split in splits.items():
        i = 0
        while i < len(split) - 1:
            pair = (split[i], split[i + 1])
            if pair in rules:
                split[i : i + 2] = [rules[pair]]
            i += 1


while len(vocabulary) < 64:
    pair_freqs = compute_pair_freqs(splits)
    create_rule(pair_freqs, rules, vocabulary)
    replace_in_splits(splits, rules)
