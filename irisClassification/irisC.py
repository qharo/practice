import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]


#READING THE DATASETS
#train_path = tf.keras.utils.get_file("iris_training.csv", "./iris_training.csv")
#test_path = tf.keras.utils.get_file("iris_training.csv", "./iris_test.csv")
#KERAS IS A MODULE INSIDE TENSORFLOW
train = pd.read_csv('iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv('iris_test.csv', names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop("Species")
test_y = test.pop("Species")

#FEATURE COLUMNS
feature_columns = []
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

#INPUT FUNCTION
def input_fn(features, labels, training=True, batch_size=256):
    #making a dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    #shuffle if training
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

#DNN CLASSIFIER
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, 
    hidden_units=[30,10],
    n_classes=3
)

classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print(eval_result)

#TO PREDICT, USE THE classifier.predict() METHOD, BY SUPPLYING AN APPROPRIATE INPUT FUNCTION