import os
import time
from svm.sksvm import SkSvm
from logistic.sklogistic import SkLogistic
from bayes.skmnb import SkMultinomialNB
from logistic.logistic import MyLogisticRegression
from util import Util

script_dir = os.path.dirname(__file__)
train_set_path = script_dir + "/../resources/train_set.txt"
validation_set_path = script_dir + "/../resources/validation_set.txt"

sklearn_models = [SkSvm(), SkLogistic(), SkMultinomialNB()]
my_models = [MyLogisticRegression()]

def run_models(models):
    for model in models:
        start = time.time()
        model.train(train_set_path)
        end = time.time()
        print("training time: " + str(end - start) + " seconds")
        start = time.time()
        print(type(model).__name__)
        model.validate(validation_set_path)
        end = time.time()
        print("validation time: " + str(end - start) + " seconds")

run_models(my_models)
run_models(sklearn_models)
