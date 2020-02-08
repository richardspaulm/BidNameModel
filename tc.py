import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import time
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

tf.logging.set_verbosity(tf.logging.ERROR)


def create_datasets(filename, index_col=0):
    data = pd.read_csv(filename, index_col=None)
    data.Labels = data.Labels.astype(int)
    
    data_len = len(data.iloc[:, 0])
    train_ratio = 0.7

    train_len = int(data_len * train_ratio)
    test_len = data_len - train_len
    data = data.reindex(np.random.permutation(data.index))
    train_df = data.head(train_len)
    test_df = data.tail(test_len)
    return train_df, test_df

print("Creating Datasets")
train_df, test_df = create_datasets("Data.csv")
print(train_df.head())
exit()


# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Labels"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Labels"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["Labels"], shuffle=False)
print("Creating Embedding")
embedded_text_feature_column = hub.text_embedding_column(
    key="String_Generated", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

print("Creating Estimator")
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    model_dir="./text_checkpoints")


print("Training Model")
estimator.train(input_fn=train_input_fn, steps=10000)
print("Evaluating")
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))



def get_predictions(estimator, input_fn):
  return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

LABELS = [
    "negative", "positive"
]

# Create a confusion matrix on training data.
with tf.Graph().as_default():
  cm = tf.confusion_matrix(train_df["Labels"], 
                           get_predictions(estimator, predict_train_input_fn))
  with tf.Session() as session:
    cm_out = session.run(cm)

# Normalize the confusion matrix so that each row sums to 1.
cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()