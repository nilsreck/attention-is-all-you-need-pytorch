from collections import defaultdict, Counter
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

word_counter = Counter()
word_counter = Counter(words)

vocabulary = []
for s in text:
    if s not in vocabulary and s != ' ':
        vocabulary.extend(s)