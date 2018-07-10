import os
import time
from svm.sksvm import SkSvm
from logistic.sklogistic import SkLogistic
from bayes.skmnb import SkMultinomialNB
from logistic.logistic import MyLogisticRegression
from util import Util

script_dir = os.path.dirname(__file__)
train_set_path = script_dir + "/../resources/imdb_train.txt"
validation_set_path = script_dir + "/../resources/imdb_validation.txt"

sklearn_models = [SkSvm(), SkLogistic(), SkMultinomialNB()]
my_models = [MyLogisticRegression()]

def run_models(models):
    for model in models:
        model.train(train_set_path)
        print(model.validate(validation_set_path))

run_models(my_models)
