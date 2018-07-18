import os
import shutil
import tensorflow as tf

###############################################################################
# Pre-define some utils
###############################################################################

def make_dataset(images_file, labels_file):
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
        return tf.one_hot(label, 10)            # int32 -> [one hot]

    images = tf.data.FixedLengthRecordDataset(
        images_file, 784, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(decode_label)

    return tf.data.Dataset.zip((images, labels))

###############################################################################
# Define, train, and evaluate a model
###############################################################################

# Hyperparameters
epochs = 10
learning_rate = 0.25
batch_size = 128
test_batch_size = 10000

# Create training and test Datasets
train_data = make_dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
test_data = make_dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
train_data = train_data.batch(batch_size)
test_data = test_data.batch(test_batch_size)

# Define training and test Iterators
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
init_train = iterator.make_initializer(train_data)
init_test = iterator.make_initializer(test_data)

# Network parameters
num_input = 784
num_hidden = 30
num_classes = 10

# Define Variables to hold network weights and biases
w1 = tf.get_variable('w1', (num_input, num_hidden), initializer=tf.truncated_normal_initializer)
w2 = tf.get_variable('w2', (num_hidden, num_classes), initializer=tf.truncated_normal_initializer)
b1 = tf.get_variable('b1', (num_hidden), initializer=tf.truncated_normal_initializer)
b2 = tf.get_variable('b2', (num_classes), initializer=tf.truncated_normal_initializer)

# Define the input layer as the next batch from the Iterator
X, Y = iterator.get_next()

# Define the rest of the network layers
hid_layer = tf.nn.relu(tf.matmul(X, w1) + b1)
out_layer = tf.matmul(hid_layer, w2) + b2

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# Define accuracy (not loss)
correct_preds = tf.equal(tf.argmax(out_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_preds))

# Add summary ops for TensorBoard
step = 0
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
summary = tf.summary.merge_all()

# Set up FileWriter for TensorBoard logging
log_dir = './models/lowlevel'
writer = tf.summary.FileWriter(log_dir)
try: shutil.rmtree(log_dir); os.makedirs(log_dir)
except: pass

# Start a Session for training and evaluation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train with mini-batches for a number of epochs
    for epoch in range(epochs):
        sess.run(init_train)

        # Iterate through train_data in mini-batches (= 1 epoch)
        while True:
            try:
                loss_val, acc_val, summary_val, _ = sess.run((loss, accuracy, summary, train_op))
                # print('Epoch {}: Loss = {:.3f}, Accuracy = {:.3f}'.format(epoch, loss_val, acc_val))
                writer.add_summary(summary_val, step)
                step += 1
            except tf.errors.OutOfRangeError:
                print('End of Epoch {}: Loss = {:.3f}, Accuracy = {:.3f}'.format(epoch, loss_val, acc_val))
                break

    # Evaluate the final network on the test set
    sess.run(init_test)
    loss_val, acc_val = sess.run((loss, accuracy))
    print('On the test set: Loss = {:.3f}, Accuracy = {:.3f}'.format(loss_val, acc_val))

