from datasets import load_dataset
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from math import log
from sklearn.metrics import accuracy_score

# Dataset
dataset = load_dataset("imdb")
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

def preprocess_text(text):
    # Punctuations
    for punct in punctuation:
        text = text.replace(punct, ' ')
    # Digits
    for i in range(0, 10):
        text = text.replace(str(i), ' ')
    # Lower Case
    text = text.lower()
    # Stop Words
    stop_words = set(stopwords.words("english"))
    replaced_words = []
    for word in text.split():
        if word not in stop_words:
            replaced_words.append(word)
    text = ' '.join(replaced_words)

    return text

# Preprocessing
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

class NaiveBayesClassifier:
    def __init__(self):
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0
        self.prior_pos = 0
        self.prior_neg = 0
        self.pos_counter = Counter()
        self.neg_counter = Counter()

    def fit(self, train_df):
        # Positive and Negative Dataframes
        pos_texts = train_df[train_df['label'] == 1]['text']
        neg_texts = train_df[train_df['label'] == 0]['text']
        # total_pos_words: Total word count in the texts with a positive label in the train_df.
        word_count = 0
        for pos_text in pos_texts:
            word_count += len(pos_text.split())
        self.total_pos_words = word_count
        # total_neg_words: Total word count in the texts with a negative label in the train_df.
        word_count = 0
        for neg_text in neg_texts:
            word_count += len(neg_text.split())
        self.total_neg_words = word_count
        # vocab_size: Total number of unique words in train_df.
        vocab_set = set()
        for text in train_df['text']:
            for word in text.split():
                vocab_set.add(word)
        self.vocab_size = len(vocab_set)
        # prior_pos: The ratio of the positive samples to the total number of samples in train_df.
        self.prior_pos = len(pos_texts) / len(train_df)
        # prior_neg: The ratio of the negative samples to the total number of samples in train_df.
        self.prior_neg = len(neg_texts) / len(train_df)
        # pos_counter: Frequency of each word belonging to the positive class in train_df.
        pos_words = []
        for pos_text in pos_texts:
            pos_words.extend(pos_text.split())
        self.pos_counter = Counter(pos_words)
        # neg_counter: Frequency of each word belonging to the negative class in train_df.
        neg_words = []
        for neg_text in neg_texts:
            neg_words.extend(neg_text.split())
        self.neg_counter = Counter(neg_words)

    def predict(self, text):
        # Preprocessing
        text = preprocess_text(text)
        # Positive Prior Probability
        log_prob_pos = log(self.prior_pos)
        # Negative Prior Probability
        log_prob_neg = log(self.prior_neg)
        for word in text.split():
            # Positive Log Probability
            log_prob_pos += log((self.pos_counter[word] + 1) / (self.total_pos_words + self.vocab_size))
            # Negative Log Probability
            log_prob_neg += log((self.neg_counter[word] + 1) / (self.total_neg_words + self.vocab_size))
        # Predicted Class
        y_predicted = 0
        if log_prob_pos > log_prob_neg:
            y_predicted = 1
        
        return (y_predicted, log_prob_pos, log_prob_neg)

# For Testing Functions
"""
nb = NaiveBayesClassifier()
nb.fit(train_df)
print(nb.total_pos_words)
print(nb.total_neg_words)
print(nb.vocab_size)
print(nb.prior_pos)
print(nb.prior_neg)
print(nb.pos_counter["great"])
print(nb.neg_counter["great"])
prediction1 = nb.predict(test_df.iloc[0]["text"])
print(f"{'Positive' if prediction1[0] == 1 else 'Negative'}")
print(prediction1)
prediction2 = nb.predict("This movie will be place at 1st in my favourite movies!")
print(f"{'Positive' if prediction2[0] == 1 else 'Negative'}")
print(prediction2)
prediction3 = nb.predict("I couldn't wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment.")
print(f"{'Positive' if prediction3[0] == 1 else 'Negative'}")
print(prediction3)
print(preprocess_text("This movie will be place at 1st in my favourite movies!"))
print(preprocess_text("I couldn’t wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment."))
y_true = test_df['label'].values
y_pred = [nb.predict(text)[0] for text in test_df['text']]
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
"""