import os
import shutil
import tensorflow as tf

###############################################################################
# Pre-define some utils
###############################################################################

def make_dataset(images_file, labels_file, batch_size, epochs=1):
    """Reads MNIST files into a Dataset.
       Identical to mlp_estimator_custom.py.
       Based on: mlp_low_level.py, with slight tweaks for Estimators."""
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
batch_size = 128                       # 1 step = 1 batch
steps_per_epoch = 60000 // batch_size  # 1 epoch = entire training set of 60000

# Network parameters
# (the input size is defined by feature_columns later down)
num_hidden = 30
num_classes = 10

# Set up Checkpoints
model_dir = './models/premade'
try: shutil.rmtree(model_dir); os.makedirs(model_dir)
except: pass

# Define the Estimator model as a DNNClassifier
estimator = tf.estimator.DNNClassifier(
    feature_columns = [tf.feature_column.numeric_column('image', shape=784)],
    hidden_units = [num_hidden],
    n_classes = num_classes,
    optimizer = tf.train.GradientDescentOptimizer(learning_rate),
    loss_reduction = tf.losses.Reduction.MEAN,
    config = tf.estimator.RunConfig(
        log_step_count_steps=steps_per_epoch,  # Log every X number of steps
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
# Higher-level train_and_evaluate(), TrainSpec, and EvalSpec
###############################################################################

# Remove Checkpoints so we can train from the start.
try: shutil.rmtree(model_dir); os.makedirs(model_dir)
except: pass

# train_and_evaluate() allows you to easily train in distributed mode (by just
# specifing the shell variable TF_CONFIG), evaluate, and export.

# However, you must explicitly specify the max number of steps to run, since it
# will evaluate after an OutOfRangeError and then start training again, looping
# forever. This is an artifact of supporting distributed training.
total_steps = epochs * steps_per_epoch

# TrainSpec is self-explanatory. EvalSpec can also include a list of Exporters.
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=total_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

###############################################################################
# Note on automatically saving and loading with Checkpoints
###############################################################################

# See mlp_custom_estimator.py

###############################################################################
# Note on exporting the model for Serving
###############################################################################

# It's actually quite hard to export an Estimator for serving. You need to
# define a serving_input_receiver_fn() that takes tf.Examples during serving
# and splices them into the Estimator graph, and then export a SavedModel with:
#   estimator.export_savedmodel(export_dir, serving_input_receiver_fn)

# There is risk of processing skew between input_fn() during training and
# serving_input_receiver_fn() during serving, since it takes extra work to keep
# the two functions in sync (is this the purpose of tf.Transform?).

# You can run the SavedModel with TF Serving with a command like:
#   tensorflow_model_server --port=9000 --model_base_path=$path_to_savedmodel

###############################################################################
# Note on TensorBoard
###############################################################################

# See mlp_custom_estimator.py

# Note that there are extra DNN metrics logged by default (such as the scalar
# metric fraction_of_zero_values, as well as histogram metrics), and that
# accuracy is not automatically logged during training.

###############################################################################
# Note on Feature Columns
###############################################################################

# See mlp_custom_estimator.py
