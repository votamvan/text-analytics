import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
import bert
from bert import tokenization
from bert import run_classifier

from sentiment.config import RESOURCE_PATH

MAX_SEQ_LENGTH = 128
label_list = [0, 1]
export_dir = RESOURCE_PATH/"export-model/colab-bert/"
vocab_file = RESOURCE_PATH/"multi_cased_L-12_H-768_A-12/vocab.txt"
predict_fn = predictor.from_saved_model(str(export_dir))
tokenizer = bert.tokenization.FullTokenizer(vocab_file=str(vocab_file))

# Convert input data into serialized Example strings.
def predict(text):
    pred_sentences = [text]
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in pred_sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    examples = []
    for item in input_features:
        feature = {}
        feature["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=item.input_ids))
        feature["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=item.input_mask))
        feature["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=item.segment_ids))
        feature["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]*MAX_SEQ_LENGTH))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(example.SerializeToString())

    # Make predictions.
    predictions = predict_fn({'examples': examples})
    print(predictions)
    prob = np.exp(predictions["probabilities"][0])
    label = predictions["labels"]
    return prob[label]