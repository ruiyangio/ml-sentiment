import os
import time
from bayes.skmnb import SkMultinomialNB
from logistic.sklogistic import SkLogistic
from svm.sksvm import SkSvm
from logistic.logistic import LogisticRegression
from util import Util

script_dir = os.path.dirname(__file__)
train_set_path = script_dir + "/../resources/train_set.txt"
validation_set_path = script_dir + "/../resources/validation_set.txt"
all_set_path = script_dir + "/../resources/all.txt"
small_set_path = script_dir + "/../resources/small_set.txt"

# model = SkLogistic(train_set_path, validation_set_path)
# model.train_and_validate()
# nb = SkMultinomialNB(1, train_set_path, validation_set_path)
# nb.train_and_validate()
# svm = SkSvm(train_set_path, validation_set_path)
# svm.train_and_validate()

t0 = time.time()
x, testX, y, testY = Util.getTrainTestFeatureFromFile(all_set_path)
model = LogisticRegression()
model.train(x, y)
t1 = time.time()
print("Train time:" + str(t1-t0))
print(model.test(testX, testY))
t2 = time.time()
print("Test time:" + str(t1-t0))
