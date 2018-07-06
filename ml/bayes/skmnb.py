from sklearn.naive_bayes import MultinomialNB
from modelbase import ModelBase

class SkMultinomialNB(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.model = MultinomialNB()
