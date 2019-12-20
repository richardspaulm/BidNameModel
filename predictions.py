import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import numpy as np
export_dir = "./training_checkpoints"

# predict_fn = predictor.from_saved_model(export_dir)


test_df = pd.read_csv("./name_training.csv", index_col=0)
test_df = test_df.drop([2151784])
test_df.Labels = test_df.Labels.astype(int)

test_df = test_df.reindex(np.random.permutation(test_df.index))

test_df = test_df.tail(100)

test_df.index = np.arange(len(test_df.iloc[:,0]))

embedded_text_feature_column = hub.text_embedding_column(
    key="String_Generated", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    model_dir="./text_checkpoints")

predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, shuffle=False)

predictions = estimator.predict(input_fn=predict_train_input_fn)

pred = [predictions.__next__() for i in range(99)]
index = 0
for p in pred:
    print(str(p), str(test_df.loc[index, "Labels"]), test_df.loc[index, "String_Generated"])
    print("\n")
    index += 1
