import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#READING THE TRAIN AND TEST SETS OF TITANIC PASSENGERS
dftrain = pd.read_csv("train.csv") 
dfeval = pd.read_csv("eval.csv")
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")
 

#CONVERTING CATEGORICAL FEATURES TO UNIQUE NUMERIC VALUES
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) 
    #previously, you would have to use a pipeline which transforms the data, but tensorflow takes care of that part ^^

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#MAKING AN INPUT FUNCTION
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, num_epochs=1)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result["accuracy"])
#ACCURACY ~74%



print(feature_columns)