import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

class Util(object):
    LABEL_SEPRATOR = "@@@@"
    LABEL_POSITIVE = "POS"
    LABEL_NEGATIVE = "NEG"

    @staticmethod
    def normalizeString(text):
        #EMOJIS
        text = re.sub(r":\)", "emojihappy1", text)
        text = re.sub(r":P", "emojihappy2", text)
        text = re.sub(r":p", "emojihappy3", text)
        text = re.sub(r":>", "emojihappy4", text)
        text = re.sub(r":3", "emojihappy5", text)
        text = re.sub(r":D", "emojihappy6", text)
        text = re.sub(r" XD ", "emojihappy7", text)
        text = re.sub(r" <3 ", "emojihappy8", text)

        text = re.sub(r":\(", "emojisad9", text)
        text = re.sub(r":<", "emojisad10", text)
        text = re.sub(r":<", "emojisad11", text)
        text = re.sub(r">:\(", "emojisad12", text)

        #MENTIONS "(@)\w+"
        text = re.sub(r"(@)\w+", "mentiontoken", text)

        #WEBSITES
        text = re.sub(r"http(s)*:(\S)*", "linktoken", text)

        #STRANGE UNICODE \x...
        text = re.sub(r"\\x(\S)*", "", text)

        #General Cleanup and Symbols
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip().lower()

    @staticmethod
    def defaultTokenize(text):
        text = Util.normalizeString(text)
        return re.split(r"\s+", text)

    @staticmethod
    def getFeatureFromFile(dataPath):
        target = []
        with open(dataPath, "r") as dataFile:
            dataContent = dataFile.readlines()

        for i, line in enumerate(dataContent):
            parts = line.split(Util.LABEL_SEPRATOR)
            s = Util.normalizeString(parts[0])
            label = parts[1].rstrip()
            if label == Util.LABEL_POSITIVE:
                target.append(1)
            else:
                target.append(0)
            dataContent[i] = s
        
        vectorizer = CountVectorizer(min_df=2, tokenizer=Util.defaultTokenize)
        tfidfTransformer = TfidfTransformer()
        countFeature = vectorizer.fit_transform(dataContent)
        tfIdfFeature = tfidfTransformer.fit_transform(countFeature)
        target = np.array(target)

        return (tfIdfFeature, target)

    @staticmethod
    def getTrainTestFeatureFromFile(dataPath, split = 0.20):
        tfIdfFeature, target = Util.getFeatureFromFile(dataPath)
        return train_test_split(tfIdfFeature, target, test_size = split, random_state = 12)
