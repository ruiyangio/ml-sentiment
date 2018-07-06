from sklearn.linear_model import LogisticRegression
from modelbase import ModelBase

class SkLogistic(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.model = LogisticRegression()
