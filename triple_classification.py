from data_processor import TKGProcessor
import numpy as np
import argparse
import sklearn as sk
import pandas as pd
import os
import pickle
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None,
                type=str,
                required=True,
                help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

parser.add_argument("--use_descriptions",
                action='store_true',
                help="Using the descriptions.")

parser.add_argument('--n_temporal_neg', type=int, default=0, metavar='N',
                               help='Number of temporal negatives.')
parser.add_argument('--n_corrupted_triple', type=int, default=0, metavar='N',
                               help='Number of non-temporal negatives.')

parser.add_argument('--min_time', type=int, default=19, metavar='N',
                 help='Minimum in time range')
parser.add_argument('--max_time', type=int, default=2020, metavar='N',
                 help='Max in time range')

parser.add_argument("--saved_features",
                action='store_true',
                help="Whether to use saved features.")

parser.add_argument("--saved_test_features",
                action='store_true',
                help="Whether to use saved test features.")

args = parser.parse_args()
processor = TKGProcessor(args.data_dir, "tc", "train", min_time=args.min_time, max_time=args.max_time)
processor.n_temporal_neg = args.n_temporal_neg
processor.n_corrupted_triple = args.n_corrupted_triple


if args.saved_features:
    with open("train_features.dat", "rb") as f:
         train_features = pickle.load(f)

else:
    train_examples = processor.get_train_examples(args.data_dir)
    train_features = processor.get_textual_features(train_examples, use_descriptions=args.use_descriptions)
    train_features = np.array(train_features)

    with open("train_features.dat", "wb") as f:
          pickle.dump(train_features, f)


X = train_features[:,:-1]
print(X.shape)
y = train_features[:,-1]
print(y.shape)

clf1 = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))\

names = [
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
]
clf1.fit(X, y)

print(round(clf1.score(X,y), 4))

lst = []
lst.append(len(processor.triple_dictionary))

processor.n_temporal_neg = 0
processor.n_corrupted_triple = 1
processor.mode = "test"

if args.saved_test_features:
    with open("test_features.dat", "rb") as f:
        test_features = pickle.load(f)
else:
    test_examples = processor.create_examples(processor._read_tsv(os.path.join(args.data_dir, "test.txt")), args.data_dir, "only_ends")
    test_features = processor.get_textual_features(test_examples, use_descriptions=args.use_descriptions)
    test_features = np.array(test_features)

    with open("test_features.dat", "wb") as f:
          pickle.dump(test_features, f)

X_test = test_features[:,:-1]
print(X_test.shape)

y_test = test_features[:,-1]
print(y_test.shape)

predicted = clf1.predict(X_test)
print("clf1 Accuracy:", metrics.accuracy_score(y_test, predicted))

lst.append(len(processor.triple_dictionary))
print(lst)

mlp = MLPClassifier(alpha=0.05, max_iter=10000)
mlp.fit(X, y)
predicted = mlp.predict(X_test)
print("MLP Accuracy:"+ str(metrics.accuracy_score(y_test, predicted)))

with open('results.txt', 'a') as result_file:
    result_file.write("MLP Accuracy:" + metrics.accuracy_score(y_test, predicted))
