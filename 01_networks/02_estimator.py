
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from os import listdir

argparser = argparse.ArgumentParser()
args = argparser.parse_args()
args.batch_size = 200
args.train_steps = 100

def input_evaluation_set(batch):
    # for now trying this on just one batch, instead of all 20
    file_directory = 'data_robot/'
    file_robot = "datawcs_robot_batch_" + batch
    file_nonrobot = "datawcs_nonrobot_batch_" + batch
    file_robot = [f for f in listdir(file_directory) if file_robot in f][0]
    file_nonrobot = [f for f in listdir(file_directory) if file_nonrobot in f][0]
    r = pd.read_pickle(file_directory + file_robot)
    nr = pd.read_pickle(file_directory + file_nonrobot)

    features = {}
    for c in r.columns[:-1]:
        features[c] = np.concatenate((r[c].values, nr[c].values), axis=0)

    labels = np.concatenate((r['label'].values, nr['label'].values), axis=0)
    return features,labels


def train_input_fn(features,labels,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)

def eval_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)



train_x, train_y = input_evaluation_set(batch='rt0')
test_x, test_y = input_evaluation_set(batch='rt1')


my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2)


classifier.train(
    input_fn = lambda:train_input_fn(train_x, train_y, args.batch_size),
    steps = args.train_steps)

eval_result = classifier.evaluate(
    input_fn = lambda:eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

