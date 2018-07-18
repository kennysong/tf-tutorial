import numpy as np
import tensorflow as tf

###############################################################################
# Pre-define some utils (these are mostly explained further down)
###############################################################################

def run_until_end(operation):
    '''Keep running an operation until an OutOfRangeError occurs,
       i.e. go through an entire Dataset.'''
    while True:
        try:
            print(sess.run(operation))
        except tf.errors.OutOfRangeError as e:
            break

def parse_row(row):
    '''Parses a row of datasets/data.csv into a features Tensor and a label
       Tensor.'''
    # TODO: Figure out how to set default_values with empty datatype Tensors
    default_values = [[0.0], [0.0], [0.0], [0]]
    parsed_row = tf.decode_csv(row, record_defaults=default_values)
    return tf.stack(parsed_row[:-1]), tf.stack(parsed_row[-1])

###############################################################################
# Creating a Numpy array Dataset, and using an Iterator
###############################################################################

# We can create a Dataset from a Numpy array of 3 rows of tuples
raw_data = np.array([[1,1],[2,2],[3,3]], dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
print(dataset.output_types)   # 'float32'
print(dataset.output_shapes)  # (2, )

# To iterate through a Dataset, we need to first wrap it in an Iterator object
iterator = dataset.make_one_shot_iterator()
get_next = iterator.get_next()

# Repeatedly run the get_next Operation in a Session to go through the Dataset
with tf.Session() as sess:
    while True:  # This loop is nasty, we wrap it into run_until_end() later
        try:
            print(sess.run(get_next))  # array([1.,1.]) ...
        except tf.errors.OutOfRangeError as e:
            print(e.message)  # 'End of sequence'
            break

# You can use an initializable Iterator to go through the Dataset multiple
# times. Child Operations of get_next will work as expected!
iterator = dataset.make_initializable_iterator()
get_next = iterator.get_next()
double = 2 * get_next

# Again, repeatedly run get_next to iterate through the Dataset (multiple times)
with tf.Session() as sess:
    for epoch in range(3):
        sess.run(iterator.initializer)
        run_until_end(double)  # array([2.,2.]) ...

###############################################################################
# Creating a Numpy array Dataset, transforming it, and using an Iterator
###############################################################################

# Repeating a Dataset, without touching the Iterator
raw_data = np.array([[1,1],[2,2],[3,3],[4,4]], dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
dataset = dataset.repeat(2)  # No argument here will loop indefinitely
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    run_until_end(get_next)  # array([1.,1.]) ...

# Batching a Dataset, which retrives multiple rows on each get_next Op
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
dataset = dataset.batch(2)
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    run_until_end(get_next)  # array([[1.,1.],[2.,2.]]) ...

# Randomizing a Dataset
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
dataset = dataset.shuffle(10)  # Buffer size to sample random elements from
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    run_until_end(get_next)  # array([3.,3.]) ... in random order

# Mapping a Dataset
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
dataset = dataset.map(lambda row: row - 1)
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    run_until_end(get_next)  # array([0.,0.]) ...

# Zipping a Dataset
raw_labels = np.arange(4)
features = tf.data.Dataset.from_tensor_slices(raw_data)
labels = tf.data.Dataset.from_tensor_slices(raw_labels)
dataset = tf.data.Dataset.zip((features, labels))
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    run_until_end(get_next)  # (array([1.,1.]), 0) ...

# Alternatively to zip, feed features and labels into a Dataset simultaneously
dataset = tf.data.Dataset.from_tensor_slices((raw_data, raw_labels))
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    run_until_end(get_next)  # (array([1.,1.]), 0) ...

###############################################################################
# Other data sources that can back Datasets
###############################################################################

# A range Dataset can be created with no data source
dataset = tf.data.Dataset.range(0, 5, 2)
get_next = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    run_until_end(get_next)  # 0 ... 2 ... 4

# A dictionary of features can also back a Dataset
raw_data = {'a':[1,2], 'b':[3,4], 'label':[0,1]}
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
get_next = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    run_until_end(get_next)  # {'a':1, 'b':3, 'label':0} ...

# TextLineDataset reads each row of a text file as a string
dataset = tf.data.TextLineDataset('./datasets/strings.txt')
get_next = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    run_until_end(get_next)  # 'string 1' ... 'string 2' ...

# TextLineDataset on a CSV of numbers, requires pre-processing
dataset = tf.data.TextLineDataset('./datasets/data.csv')
dataset = dataset.skip(1)
dataset = dataset.map(parse_row)
get_next = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    run_until_end(get_next)  # (array([1.,2.,3.]), 0) ...

# FixedLengthRecordDataset reads a certain number of bytes per row, from a
# binary file (such as in image datasets).
# dataset = tf.data.FixedLengthRecordDataset('filename', bytes_per_record)
# ...rest skipped

# TFRecordDataset reads from a .tfrecord file, which is "a simple record-
# oriented binary format"
# dataset = tf.data.TFRecordDataset('filename.tfrecord')
# ...rest skipped

###############################################################################
# Types of Iterators that can switch Datasets
###############################################################################

# An initializable Iterator can parameterized by a Placeholder.
# We parameterize range() here, but can also parameterize from_tensor_slices()
# with Placeholders of the same sizes, and feed in arrays at Session runtime.
# Otherwise, from_tensor_slices() will store the data as Constants in the
# Graph, potentially causing bloat.
m, k = tf.placeholder(tf.int64, shape=()), tf.placeholder(tf.int64, shape=())
dataset = tf.data.Dataset.range(m)
iterator = dataset.make_initializable_iterator()
added = iterator.get_next() + k

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={m: 5})  # This is only run once
    for _ in range(5):
        print(sess.run(added, feed_dict={k: -1}))  # -1 ... 0 ... 1 ...

# A re-initializable Iterator can iterate over different Datasets with the same
# structures.
# This is useful for running the same computational Graph over a training set,
# and then a validation set. Otherwise, each Iterator would have its own
# computational Graph (and it would be difficult to share parameter Variables).
dataset_A = tf.data.Dataset.range(5)
dataset_B = tf.data.Dataset.range(5, 10)
iterator = tf.data.Iterator.from_structure(dataset_A.output_types,
                                           dataset_A.output_shapes)
init_A = iterator.make_initializer(dataset_A)
init_B = iterator.make_initializer(dataset_B)
get_next = iterator.get_next()

with tf.Session() as sess:
    sess.run(init_A)
    run_until_end(get_next)  # 0 ... 1 ... 2 ...
    sess.run(init_B)
    run_until_end(get_next)  # 4 ... 5 ... 6 ...

# A feedable Iterator is like a re-initializable one, but wraps other Iterator
# from its string handle. Its benefit is that you don't have to initialize the
# Iterator, i.e. reset its position in the Dataset, when switching Datasets.
# But this is a weird pattern, so only a general outline is given.
# handle = tf.placeholder(tf.string, shape=[])
# iterator = tf.data.Iterator.from_string_handle(handle, output_types, output_shapes)
# iterator_A = datasetA.make_one_shot_iterator()
# iterator_B = datasetB.make_initializable_iterator()
# handle_A = sess.run(iterator_A.string_handle())
# ...
# sess.run(iterator.get_next(), feed_dict={handle: handle_A})
# sess.run(iterator.get_next(), feed_dict={handle: handle_B})
