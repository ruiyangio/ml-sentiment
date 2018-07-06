import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from util import Util

class FeatureExtractor(object):
    LABEL_SEPRATOR = "@@@@"
    LABEL_POSITIVE = "POS"
    LABEL_NEGATIVE = "NEG"

    @staticmethod
    def train_test_split(x, y, split = 0.8):
        return train_test_split(x, y, test_size = split, random_state = 12)

    def __init__(self, ngram_range=(1,2)):
        self.vectorizer = CountVectorizer(min_df=2, tokenizer=Util.default_tokenize, ngram_range=ngram_range)
        self.tfidf_transformer = TfidfTransformer()

    def fit_transform(self, data_path, fit = True):
        target = []
        with open(data_path, "r") as data_file:
            data_content = data_file.readlines()

        for i, line in enumerate(data_content):
            parts = line.split(self.LABEL_SEPRATOR)
            s = parts[0]
            label = parts[1].rstrip()
            if label == self.LABEL_POSITIVE:
                target.append(1)
            else:
                target.append(0)
            data_content[i] = s
        count_feature = self.vectorizer.fit_transform(data_content) if fit else self.vectorizer.transform(data_content)
        tfidf = self.tfidf_transformer.fit_transform(count_feature) if fit else self.tfidf_transformer.transform(count_feature)

        return (tfidf, np.array(target))

    def transform(self, data_path):
        return self.fit_transform(data_path, False)
