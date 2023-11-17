import numpy as np
import tensorflow as tf
import pandas as pd
# import datalib
import os

def load_dataset(data, path_to_data_dir='data'):
    if data == 'student':
        label = 'final_result'
    elif data == 'taiwan':
        label = 'Y'
    else:
        label = 'label'

    train = pd.read_csv(os.path.join(path_to_data_dir, data) + '_train.csv')
    test = pd.read_csv(os.path.join(path_to_data_dir, data) + '_test.csv')
    y_train = np.array(train[label]).astype('float32')
    y_test = np.array(test[label]).astype('float32')
    X_train = np.array(train.drop([label], axis=1)).astype('float32')
    X_test = np.array(test.drop([label], axis=1)).astype('float32')
    n_classes = 2

    return (X_train, y_train), (X_test, y_test), n_classes


def batch_flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def convert_to_super_labels(preds, affinity_set):
    for subset in affinity_set:
        for l in subset:
            preds[preds == l] = subset[0]
    return preds

def invalidation(counterfactuals,
                 modelA_counterfual_preds,
                 modelB,
                 modelA_pred=None,
                 batch_size=32,
                 affinity_set=[[0], [1, 2]]):

    if modelA_pred is None:
        modelA_pred = 1 - modelA_counterfual_preds

    modelB_counterfactual_probits = modelB.predict(counterfactuals,
                                                   batch_size=batch_size)

    is_bianary = modelB_counterfactual_probits.shape[1] <= 2

    modelB_counterfactual_pred = np.argmax(modelB_counterfactual_probits,
                                           axis=-1)
    
    if is_bianary:
        validation = np.mean(
            modelA_pred != modelB_counterfactual_pred)
    else:
        modelA_super_labels = convert_to_super_labels(modelA_pred.copy(), affinity_set)
        modelB_counterfactual_super_labels = convert_to_super_labels(modelB_counterfactual_pred.copy(), affinity_set)

        validation = np.mean(
            modelA_super_labels != modelB_counterfactual_super_labels)


    return 1.0 - validation,