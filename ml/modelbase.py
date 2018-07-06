from sklearn.metrics import accuracy_score
from extractor import FeatureExtractor

class ModelBase(object):
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = None

    def train(self, data_path):
        x, y = self.feature_extractor.fit_transform(data_path)
        self.model.fit(x, y)

    def validate(self, data_path):
        x, y = self.feature_extractor.transform(data_path)
        y_pred = self.model.predict(x)
        return accuracy_score(y, y_pred)
