import numpy as np
import math
from util import Util

class MultinomialNB():
    def __init__(self):
        self.vocabulary = set()
        self.vocabularyCount = 0
        self.totalDocuments = 0
        self.categories = set()
        self.categoriesDocuments = {}
        self.categoriesTokenCounts = {}
        self.categoriesVocabulary = {}

    def getTokenFrequency(self, tokens):
        res = {}
        for token in tokens:
            res[token] = 1 if token not in res else res[token] + 1
        return res

    def getLikelihood(self, token, category):
        tokenCount = self.categoriesVocabulary[category][token] if token in self.categoriesVocabulary[category] else 0
        totalCategoryWords = self.categoriesTokenCounts[category]
        return np.log( ( tokenCount + 1 ) / ( totalCategoryWords + self.vocabularyCount ) )

    def tokenize(self, text):
        return Util.defaultTokenize(text)

    def train(self, text, category):
        if category not in self.categories:
            self.categories.add(category)
            self.categoriesDocuments[category] = 0
            self.categoriesTokenCounts[category] = 0
            self.categoriesVocabulary[category] = {}

        self.totalDocuments += 1
        self.categoriesDocuments[category] += 1

        tokens = self.tokenize(text)
        textTokensFrequency = self.getTokenFrequency(tokens)

        for token in textTokensFrequency:
            if token not in self.vocabulary:
                self.vocabulary.add(token)
                self.vocabularyCount += 1

            if token not in self.categoriesVocabulary[category]:
                self.categoriesVocabulary[category][token] = textTokensFrequency[token]
            else:
                self.categoriesVocabulary[category][token] += textTokensFrequency[token]

            self.categoriesTokenCounts[category] += textTokensFrequency[token]

    def predict(self, text):
        resCategory = None
        maxProb = -math.inf

        tokens = self.tokenize(text)
        textTokensFrequency = self.getTokenFrequency(tokens)

        for category in self.categories:
            currProb = np.log(self.categoriesDocuments[category] / self.totalDocuments)

            for token in tokens:
                currProb += textTokensFrequency[token] * self.getLikelihood(token, category)

            if currProb > maxProb:
                maxProb = currProb
                resCategory = category

        return resCategory
