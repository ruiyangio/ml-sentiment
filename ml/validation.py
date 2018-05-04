import os
from bayes.mnb import MultinomialNB

script_dir = os.path.dirname(__file__)

nb = MultinomialNB()
with open(script_dir + "/../resources/train_set.txt") as train_file:
    line = train_file.readline()
    while line:
        parts = line.split("@@@@")
        nb.train(parts[0], parts[1])
        line = train_file.readline()

print("Training completed. Start validating...")
correct = 0
total = 0
with open(script_dir + "/../resources/validation_set.txt") as validation_file:
    line = validation_file.readline()
    while line:
        total += 1
        parts = line.split("@@@@")
        if parts[1] == nb.predict(parts[0]):
            correct += 1
        line = validation_file.readline()

print(correct / total)