import os
from bayes.skmnb import SkMultinomialNB
from logistic.sklogistic import SkLogistic
from svm.sksvm import SkSvm

script_dir = os.path.dirname(__file__)
train_set_path = script_dir + "/../resources/train_set.txt"
validation_set_path = script_dir + "/../resources/validation_set.txt"

# model = SkLogistic(train_set_path, validation_set_path)
# model.train_and_validate()
# nb = SkMultinomialNB(1, train_set_path, validation_set_path)
# nb.train_and_validate()
svm = SkSvm(train_set_path, validation_set_path)
svm.train_and_validate()