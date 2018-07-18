import os
import shutil
import tensorflow as tf

# Before we do anything, clear the log folder
log_dir = './logs'
try: shutil.rmtree(log_dir); os.makedirs(log_dir)
except: pass

# Adding some Tensors to the Graph
a = tf.constant(5.0)
b = tf.ones((2,4))  # This is also a Const Operation
p = tf.placeholder(tf.float32, shape=())
q = tf.placeholder(tf.float32)

# Adding some Tensor Operations to the Graph
c = a + b
d = c + p * a

# Adding some Variables to the Graph
u = tf.Variable(3.0)
v = tf.get_variable('v', (1))

# Adding some Variable Operations to the Graph
w = tf.add(u, v, name='w')

# Adding a Variable scope to the Graph
with tf.variable_scope('scoped'):
    x = tf.get_variable('x', shape=(2,4))
    y = tf.get_variable('y', shape=(4,4))
    result = tf.matmul(x, y, name='result')

# Add a scoped Operation to the Graph
z = result + (x + b)

# Write the Graph to a log file
writer = tf.summary.FileWriter('./logs')
writer.add_graph(tf.get_default_graph())

# Write "timestamped" scalar values to a log file
t = tf.Variable(2.0)
tf.summary.scalar('t', t)         # Tensor('t:0', dtype=string), this tracks the value of the Variable at t
summary = tf.summary.merge_all()  # This concats all summary Tensors (in GraphKeys.SUMMARIES collection) into one Tensor

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        _, summary_val = sess.run((t.assign(t+1), summary))  # summary_val is the serialized value of the Tensor that summary 't' tracks
        writer.add_summary(summary_val, step)                # Write the string summary_val to an events log file
        writer.flush()                                       # This is necessary since this loop is too fast

# Run this in another terminal to start the TensorBoard server
# tensorboard --logdir logs --host 127.0.0.1

