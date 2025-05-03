from datasets import load_dataset
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

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

def bias_scores(train_df):
    # Positive and Negative Dataframes
    pos_texts = train_df[train_df['label'] == 1]['text']
    neg_texts = train_df[train_df['label'] == 0]['text']
    # Positive Word Counter
    pos_words = []
    for pos_text in pos_texts:
        pos_words.extend(pos_text.split())
    pos_counter = Counter(pos_words)
    # Negative Word Counter
    neg_words = []
    for neg_text in neg_texts:
            neg_words.extend(neg_text.split())
    neg_counter = Counter(neg_words)
    # Bias Score of Words
    bias_scores = []
    vocab = set(pos_counter.keys()).union(neg_counter.keys())
    for text in vocab:
        for word in text.split():
            fp = pos_counter[word]
            fn = neg_counter[word]
            ft = fp + fn
            score = abs((fp - fn) / ft) * log(ft)
            bias_scores.append((word, fp, fn, ft, score))
    # Keeping only the top 10,000
    bias_scores.sort(key=lambda x: (-x[4], x[0]))
    top10k_bias_scores = bias_scores[:10000]

    return top10k_bias_scores

# Top 10000 Highest Bias Scores
scores = bias_scores(train_df)

#For Testing Function
"""
print(scores[:2])
print(scores[-2:])
"""

# Bag-of-Words
top10k_words = []
for score in scores:
    top10k_words.append(score[0])
bag_of_words = CountVectorizer(vocabulary=top10k_words)

# Feature Matrices for Training and Test Datasets
X_train = bag_of_words.transform(train_df['text'])
X_test = bag_of_words.transform(test_df['text'])

# Feature Matrices in X_train and X_test
y_train = train_df['label']
y_test = test_df['label']

# Accuracy Score Dicts
train_accuracy_scores = dict()
test_accuracy_scores = dict()

# Training
for i in range(1, 26):
    # Creating Model
    lr_model = LogisticRegression(max_iter=i)
    lr_model.fit(X_train, y_train)
    # Prediction Results
    prediction_train = lr_model.predict(X_train)
    prediction_test = lr_model.predict(X_test)
    # Accuracy Scores
    acc_train = accuracy_score(y_train, prediction_train)
    acc_test = accuracy_score(y_test, prediction_test)
    # Adding to Accuracy Score Dicts
    train_accuracy_scores[i] = acc_train
    test_accuracy_scores[i] = acc_test

# Graph
plt.plot(train_accuracy_scores.keys(), train_accuracy_scores.values(), label="Train", marker='o')
plt.plot(test_accuracy_scores.keys(), test_accuracy_scores.values(), label="Test", marker='o')
plt.xticks([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25])
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy Score")
plt.title("Train vs Test Accuracy Scores for Logistic Regression")
plt.legend()
plt.show()

# Small Analysis
"""
By looking at the graph, we see that test accuracy scores increased until certain point, and then stayed almost stationary until the end eventhough train kept increasing.
We need to use the model where train and test accuracy scores are near to each other and test score is almost the highest.
I would use the 11th model, as it is where test accuracy scores reach its likely maximum and there is no certain difference between train and test yet.
This way, the optimal logistic regression classifier would be reached.
"""