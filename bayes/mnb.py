import numpy as np
import math
import re

class MultinomialNB():
    CLEAN_PATTERN = re.compile(r"[^(a-zA-Z0-9_)+\s]")
    SPLIT_PATTERN = re.compile(r"\s+")
    
    def __init__(self):
        self.vocab = set()
        self.totalDocuments = 0
        self.categories = {}

    def getTokenFrequency(self, tokens):
        res = {}
        for token in tokens:
            res[token] = 1 if token not in res else res[token] + 1
        return res

    def getLikelihood(self, token, category):
        tokenCount = self.categories[category]["tokenFrequency"][token] if token in self.categories[category]["tokenFrequency"] else 0
        totalCategoryWords = self.categories[category]["totalTokenCount"]
        return np.log( ( tokenCount + 1 ) / ( totalCategoryWords + len(self.vocab) ) )

    def tokenize(self, text):
        text = text.rstrip().lower()
        text = self.CLEAN_PATTERN.sub(" ", text)
        return self.SPLIT_PATTERN.split(text)

    def train(self, text, category):
        if category not in self.categories:
            self.categories[category] = { "documents": 0, "totalTokenCount": 0, "tokenFrequency": {} }
        
        self.totalDocuments += 1
        self.categories[category]["documents"] += 1

        tokens = self.tokenize(text)
        textTokensFrequency = self.getTokenFrequency(tokens)

        for token in textTokensFrequency:
            if token not in self.vocab:
                self.vocab.add(token)
            if token not in self.categories[category]["tokenFrequency"]:
                self.categories[category]["tokenFrequency"][token] = textTokensFrequency[token]
            else:
                self.categories[category]["tokenFrequency"][token] += textTokensFrequency[token]
            
            self.categories[category]["totalTokenCount"] += textTokensFrequency[token]

    def predict(self, text):
        resCategory = None
        maxProb = -math.inf

        tokens = self.tokenize(text)
        textTokensFrequency = self.getTokenFrequency(tokens)

        for category in self.categories:
            currProb = np.log(self.categories[category]["documents"] / self.totalDocuments)
            
            for token in tokens:
                likelihood = self.getLikelihood(token, category)
                currFrequncy = textTokensFrequency[token]
                currProb += currFrequncy * likelihood

            if currProb > maxProb:
                maxProb = currProb
                resCategory = category

        return resCategory

nb = MultinomialNB()
with open("./resources/train_set.txt") as train_file:
    line = train_file.readline()
    while line:
        parts = line.split("@@@@")
        nb.train(parts[0], parts[1])
        line = train_file.readline()

print("Training completed. Start validating...")
print(len(nb.vocab))
correct = 0
total = 0
with open("./resources/validation_set.txt") as validation_file:
    line = validation_file.readline()
    while line:
        total += 1
        parts = line.split("@@@@")
        if parts[1] == nb.predict(parts[0]):
            correct += 1
        line = validation_file.readline()

print(correct / total)