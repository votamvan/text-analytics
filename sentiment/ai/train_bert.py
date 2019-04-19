from datetime import datetime
from pathlib import Path
import pandas as pd
import tensorflow as tf

import bert
from bert import tokenization
from bert import run_classifier
from bert import modeling
from bert import optimization

from sentiment.config import RESOURCE_PATH

OUTPUT_DIR = str(RESOURCE_PATH/'OUTPUT-BERT')
tf.gfile.MakeDirs(OUTPUT_DIR)
BERT_MODEL_HUB = RESOURCE_PATH/"multi_cased_L-12_H-768_A-12/"

flags = tf.flags
FLAGS = flags.FLAGS
FLAGS.output_dir = OUTPUT_DIR
FLAGS.vocab_file = str(RESOURCE_PATH/"multi_cased_L-12_H-768_A-12/vocab.txt")
FLAGS.bert_config_file = str(RESOURCE_PATH/"multi_cased_L-12_H-768_A-12/bert_config.json")

def load_datasets(data_path):
    _path = Path(data_path)
    data_vi = pd.read_csv(_path/"train.csv")
    data_en = pd.read_csv(_path/"imdb-train.csv", names=['label', 'comment'])
    data = data_vi.append(data_en, ignore_index=True).sample(frac=1)
    total_rows = data.shape[0]
    split_index = int(total_rows / 4.0)
    test = data[:split_index]
    train = data[split_index:]
    return train, test

data_path = RESOURCE_PATH/"data/"
train, test = load_datasets(data_path)
# max_sample = 32
# train = train.sample(max_sample)
# test = test.sample(max_sample)

DATA_COLUMN = 'comment'
LABEL_COLUMN = 'label'
label_list = [0, 1]

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

tokenizer = bert.tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
tokens = tokenizer.tokenize("Nhà thờ Đức Bà Paris, được bắt đầu xây dựng từ năm 1163 và hoàn thành vào năm 1345.")
print(tokens)

MAX_SEQ_LENGTH = 128
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


def create_model(bert_config, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings=False):
  """Creates a classification model."""
  is_training = not is_predicting
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)
  
# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(bert_config, num_labels, learning_rate, num_train_steps, num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    print("input_ids", input_ids.shape)
    print("input_mask", input_mask.shape)
    print("segment_ids", segment_ids.shape)
    print("label_ids", label_ids.shape)

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(bert_config,
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
        auc = tf.metrics.auc(label_ids, predicted_labels)
        recall = tf.metrics.recall(label_ids, predicted_labels)
        precision = tf.metrics.precision(label_ids, predicted_labels)
        true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
        true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
        false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
        false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)
      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(bert_config,
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn



# Compute train and warmup steps from batch size
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS
)

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
model_fn = model_fn_builder(
  bert_config=bert_config,
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps
)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE}
)

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
  features=train_features,
  seq_length=MAX_SEQ_LENGTH,
  is_training=True,
  drop_remainder=False
)

print('========== Beginning Training! ==========')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("========== Training took time ==========", datetime.now() - current_time)

test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

evals = estimator.evaluate(input_fn=test_input_fn, steps=None)
print("========== Evaluation ==========", evals)


def getPrediction(in_sentences):
  labels = ["Positive", "Negative"]
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!",
  "Xin lỗi anh"
]
predictions = getPrediction(pred_sentences)
print(predictions)


print("========== Save model ==========")
keys = ["input_ids", "input_mask", "segment_ids", "label_ids"]
feature_columns = [tf.feature_column.numeric_column(key=key, shape=MAX_SEQ_LENGTH, dtype=tf.int64) for key in keys]
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = estimator.export_savedmodel(
    export_dir_base="./export-model/",
    serving_input_receiver_fn=serving_input_receiver_fn
)
print("export_dir", export_dir)

print("========== Predict ==========")
from tensorflow.contrib import predictor
predict_fn = predictor.from_saved_model(export_dir)
# Convert input data into serialized Example strings.
input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in pred_sentences] # here, "" is just a dummy label
input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
keys = ["input_ids", "input_mask", "segment_ids", "label_ids"]
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