from collections import Counter
import string

corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning.",
]


class CustomBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocabulary = []
        self.rules = {}
        self.word_counter = None
        self.splits = None

    def fit(self, corpus):
        text = " ".join(corpus)
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        self.word_counter = Counter(words)

        for s in text:
            if s not in self.vocabulary and s != " ":
                self.vocabulary.extend(s)
        self.splits = {word: [c for c in word] for word in self.word_counter.keys()}

        while len(self.vocabulary) < 64:
            pair_freqs = self._compute_pair_freqs(self.splits)
            self._create_rule(pair_freqs)
            self._replace_in_splits()

    def encode(self, text):
        encoding = []
        split_seq = text.split()
        sorted_vocab = sorted(self.vocabulary, key=len, reverse=True)

        for word in split_seq:
            while word:
                matched = False
                for entry in sorted_vocab:
                    if word.startswith(entry):
                        encoding.append(entry)
                        word = word[len(entry) :]
                        matched = True
                        break
                if not matched:
                    encoding.append("<UNMATCHED>")
                    break
        return encoding

    def _compute_pair_freqs(self, splits):
        pair_freqs = Counter()
        for word, freq in self.word_counter.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _create_rule(self, pair_freqs):
        most_common_pair = pair_freqs.most_common(1)[0]
        concatenated = "".join(most_common_pair[0])
        self.rules[most_common_pair[0]] = concatenated
        self.vocabulary.append(concatenated)
        pair_freqs.pop(most_common_pair[0])

    def _replace_in_splits(self):
        for _, split in self.splits.items():
            i = 0
            while i < len(split) - 1:
                pair = (split[i], split[i + 1])
                if pair in self.rules:
                    split[i : i + 2] = [self.rules[pair]]
                i += 1
