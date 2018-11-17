#!/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.estimator import DNNClassifier

def training_data():
    data = {}
    data['feature'] = []
    data['label'] = []
    with tf.gfile.GFile('training_data.txt', 'r') as f:
        feature = True
        for line in f:
            if feature:
                data['feature'].append(np.asarray(list(map(float, line.split(' ')))))
            else:
                data['label'].append(np.argmax(list(map(float, line.split(' ')))))
            feature = not feature
    return tf.estimator.inputs.numpy_input_fn(
        x={'feature': np.asarray(data['feature'])},
        y=np.asarray(data['label']),
        num_epochs=None,
        batch_size=50,
        shuffle=True)

def test_data():
    data = {}
    data['feature'] = []
    data['label'] = []
    with tf.gfile.GFile('test_data.txt', 'r') as f:
        feature = True
        for line in f:
            if feature:
                data['feature'].append(np.asarray(list(map(float, line.split(' ')))))
            else:
                data['label'].append(int(line))
            feature = not feature
    return tf.estimator.inputs.numpy_input_fn(
        x={'feature': np.asarray(data['feature'])},
        y=np.asarray(data['label']),
        num_epochs=1,
        shuffle=False)


estimator = DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column('feature', shape=[28, 28])],
    hidden_units=[100],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1
)

estimator.train(input_fn=training_data(), steps=100000)
print(estimator.evaluate(test_data()))

