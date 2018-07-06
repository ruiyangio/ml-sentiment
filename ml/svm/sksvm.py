from sklearn.svm import LinearSVC
from modelbase import ModelBase

class SkSvm(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.model = LinearSVC()
