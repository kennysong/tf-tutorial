import os
import shutil
import tensorflow as tf

###############################################################################
# Pre-define some utils
###############################################################################

def make_dataset(images_file, labels_file, batch_size, epochs=1):
    """Reads MNIST files into a Dataset.
       Based on: https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py"""
    images_file = os.path.join('datasets', images_file)
    labels_file = os.path.join('datasets', labels_file)

    def decode_image(image):
        image = tf.decode_raw(image, tf.uint8)  # String -> [uint8 bytes]
        image = tf.to_float(image)              # [uint8 bytes] -> [floats]
        image = tf.reshape(image, [784])        # This is actually redundant
        return image / 255.0                    # Normalize from [0, 255] to [0.0, 1.0]

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # String -> [uint8 byte]
        label = tf.to_int32(label)              # [uint8 byte] -> [int32]
        label = tf.reshape(label, [])           # [int32] -> int32
        return label                            # This is NOT a one-hot vector

    images = tf.data.FixedLengthRecordDataset(
        images_file, 784, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(decode_label)

    dataset = tf.data.Dataset.zip(({'image': images}, labels))
    return dataset.batch(batch_size).repeat(epochs)  # You can also return an Iterator.get_next() here

###############################################################################
# Define the model
###############################################################################

# Hyperparameters
epochs = 10
learning_rate = 0.25
batch_size = 128
steps_per_epoch = 60000 // batch_size

# Network parameters
num_input = 784
num_hidden = 30
num_classes = 10

# Set up Checkpoints
model_dir = './models/custom_layers'
try: shutil.rmtree(model_dir); os.makedirs(model_dir)
except: pass

# Define the network as a model_fn
def model_fn(features, labels, mode, params):
    # Construct the network (with the layers API)
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.layers.dense(net, units=params['num_hidden'], activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['num_classes'], activation=None)

    # Construct the network (with the low-level API)
    # Note: this is closest to mlp_low_level.py, but the layers API gets better performance (why?)
    # w1 = tf.get_variable('w1', (params['num_input'], params['num_hidden']), initializer=tf.truncated_normal_initializer)
    # w2 = tf.get_variable('w2', (params['num_hidden'], params['num_classes']), initializer=tf.truncated_normal_initializer)
    # b1 = tf.get_variable('b1', (params['num_hidden']), initializer=tf.truncated_normal_initializer)
    # b2 = tf.get_variable('b2', (params['num_classes']), initializer=tf.truncated_normal_initializer)
    # input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    # hid_layer = tf.nn.relu(tf.matmul(input_layer, w1) + b1)
    # logits = tf.matmul(hid_layer, w2) + b2

    # Mode for training the model
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = params['optimizer']
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        predicted_classes = tf.argmax(logits, 1)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
        tf.summary.scalar('accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Mode for evaluating the model
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        predicted_classes = tf.argmax(logits, 1)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    # Mode for prediction with the model
    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_classes = tf.argmax(logits, 1)
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

# Construct an Estimator, primarily a runtime wrapper around a model_fn
estimator = tf.estimator.Estimator(
    model_fn = model_fn,
    params = {
        'feature_columns': [tf.feature_column.numeric_column('image', shape=784)],
        'num_input': num_input,
        'num_hidden': num_hidden,
        'num_classes': num_classes,
        'optimizer': tf.train.GradientDescentOptimizer(learning_rate)
    },
    config = tf.estimator.RunConfig(
        log_step_count_steps=steps_per_epoch,
        model_dir=model_dir
    )
)

###############################################################################
# Train, evaluate, and predict with the model
###############################################################################

# Train the Estimator by calling .train() and passing in a Dataset-generating input_fn
train_input_fn = lambda: make_dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', batch_size, epochs)
estimator.train(train_input_fn)

# Evaluate the Estimator by calling .evaluate() and passing in a Dataset-generating input_fn
test_input_fn = lambda: make_dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', batch_size)
print(estimator.evaluate(train_input_fn))  # Not sure why this loss is not identical to the final training step
print(estimator.evaluate(test_input_fn))

# Predict with the Estimator by calling .evaluate() and passing in a Dataset-generating input_fn
predictions = estimator.predict(test_input_fn)
print(next(predictions))

###############################################################################
# Note on automatically saving and loading with Checkpoints
###############################################################################

# The Estimator model is automatically saved at certain Checkpoint intervals
# during training, in the RunConfig.model_dir directory. You can specify the
# intervals with RunConfig.save_checkpoints_secs.

# During Estimator training, evaluation, and prediction, it automatically
# restores from the latest Checkpoint in the RunConfig.model_dir directory.

###############################################################################
# Note on exporting the model for Serving
###############################################################################

# See mlp_estimator_premade.py

# For a custom Estimator, you have to specifically define the outputs in the
# model_fn's tf.estimator.EstimatorSpec return value. The output is of type
# tf.estimator.export.ClassificationOutput (classes & scores), RegressionOutput
# (a single number), or PredictOutput (raw Tensor output).

###############################################################################
# Note on TensorBoard
###############################################################################

# View TensorBoard after running this script with:
# tensorboard --logdir models --host 127.0.0.1

# Only loss and global_step/sec are automatically logged for custom Estimators.
# We also log accuracy during training and evaluations with the lines,
#   tf.summary.scalar('accuracy', accuracy[1])
#   eval_metric_ops={'accuracy': accuracy}
# respectively.

###############################################################################
# Note on Feature Columns
###############################################################################

# Feature Columns define the input layer of an Estimator model, i.e. a basic
# schema of how to read the Dataset into features. It specifies names and types
# of (groups of) Dataset columns, selects which ones to feed into the
# model, and applies a few common transformations on the raw Dataset columns.

# Think of a Dataset as a low-level way to decode raw data on disk into data
# columns in memory (e.g. decoding binary image files into vectors), and
# Feature Columns as high-level transformations from data columns
# to features (e.g. 784 Dataset columns => one "image" feature, or turning a
# vocabulary into one-hot vectors or embeddings).

# numeric_column: numbers -> numbers
# bucketized_column: numeric_column + bucket ranges -> one-hot vectors
# categorical_column_with_identity: {0, 1, 2} -> {0, 1, 2}  (this is just a validation)
# categorical_column_with_vocabulary_list: {dog, cat} -> {0, 1}
# categorical_column_with_vocabulary_file: {dog, cat} -> {0, 1}
# categorical_column_with_hash_bucket: {dog, cat, ...} -> {0, 1, ..., N} hashed into buckets
# indicator_column: categorical_column -> one-hot vectors
# embedding_column: categorical_column -> embedding vectors
# crossed_column: featureA + featureB -> featureC (e.g. lat, lon)
