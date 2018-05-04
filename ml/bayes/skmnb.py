import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class MyMultinomialNB(object):
    LABEL_SEPRATOR = "@@@@"
    LABEL_POSITIVE = "pos"
    LABEL_NEGATIVE = "neg"

    def __init__(self, alpha, train_file_path, test_file_path):
        self.dict = {}
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.alpha = alpha
        self.review_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
        self.tfidf_transformer = TfidfTransformer()
        self.clf = None

    def train_and_validate(self):
        target = []
        with open(self.train_file_path, "r") as train_file:
            train_content = train_file.readlines()

        for i, line in enumerate(train_content):
            line = line.rstrip().lower()
            parts = line.split(self.LABEL_SEPRATOR)
            s = parts[0]
            label = parts[1]

            if label == self.LABEL_POSITIVE:
                target.append(1)
            else:
                target.append(0)
            train_content[i] = s
        
        train_counts = self.review_vec.fit_transform(train_content)
        train_tfidf = self.tfidf_transformer.fit_transform(train_counts)
        docs_train, docs_test, y_train, y_test = train_test_split(train_tfidf, target, test_size = 0.40, random_state = 12)
        self.clf = MultinomialNB().fit(docs_train, y_train)
        y_pred = self.clf.predict(docs_test)
        print(sklearn.metrics.accuracy_score(y_test, y_pred))

    def train(self):
        target = []
        with open(self.train_file_path, "r") as train_file:
            train_content = train_file.readlines()

        for i, line in enumerate(train_content):
            line = line.rstrip().lower()
            parts = line.split(self.LABEL_SEPRATOR)
            s = parts[0]
            label = parts[1]

            if label == self.LABEL_POSITIVE:
                target.append(1)
            else:
                target.append(0)
            train_content[i] = s
        
        train_counts = self.review_vec.fit_transform(train_content)
        train_tfidf = self.tfidf_transformer.fit_transform(train_counts)
        self.clf = MultinomialNB().fit(train_tfidf, target)

    def validate(self):
        with open(self.test_file_path, "r") as test_file:
            test_content = test_file.readlines()

        correct = 0
        total = 0

        for line in test_content:
            total += 1
            if self.predict(line):
                correct += 1

        return correct / total

    def predict(self, line):
        line = line.rstrip().lower()
        parts = line.split(self.LABEL_SEPRATOR)
        s = parts[0]
        label = parts[1]

        line_s = [s]
        line_counts = self.review_vec.transform(line_s)
        line_tfidf = self.tfidf_transformer.transform(line_counts)
        pred_res = self.LABEL_POSITIVE if self.clf.predict(line_tfidf) == 1 else self.LABEL_NEGATIVE
        return pred_res == label

nb = MyMultinomialNB(1, "../resources/train_set.txt", "../resources/validation_set.txt")
nb.train()
print(nb.validate())
