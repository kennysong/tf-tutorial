import numpy as np
import tensorflow as tf

# Note: All ops and tensors are automatically added to the default graph

###############################################################################
# Fun with Constants (static Tensors)
###############################################################################

a = tf.constant(5)              # a.shape is ()
b = tf.constant([1,2,3])        # b.shape is (3, )
c = tf.constant([[1,2,3]])      # c.shape is (1, 3)
d = tf.constant([[1],[2],[3]])  # d.shape is (3, 1)

# Do some Tensor operations
with tf.Session() as sess:
    print(a)                                 # Tensor('Const:0')
    print(sess.run((a, b)))                  # (5, array([1,2,3]))
    print(sess.run(a+b))                     # array([6,7,8])
    print(sess.run(tf.matmul(c, d)))         # array([[14]])
    print(sess.run(tf.matmul(d, c)))         # array([[1,2,3],[2,4,6],[3,6,9]])
    print(sess.run(d[2, 0]))                 # 3
    print(sess.run(d[:, 0]))                 # array([1,2,3])
    print(sess.run(tf.reshape(b, (3, 1))))   # array([[1],[2],[3]])
    print(sess.run(tf.reshape(b, (-1, 1))))  # array([[1],[2],[3]])

###############################################################################
# Fun with Placeholders (empty Tensors)
###############################################################################

tf.reset_default_graph()
p = tf.placeholder(tf.int32, shape=())         # p must be a scalar
q = tf.placeholder(tf.int32, shape=(None, 1))  # q can be any # of rows of a 1-tuple
r = tf.placeholder(tf.int32)                   # r can be of any shape

# Feeding some empty Tensors
with tf.Session() as sess:
    print(p)                                            # Tensor('Placeholder:0')
    print(sess.run(p*r, feed_dict={p:2, r:[1,2]}))      # array([2,4])
    print(sess.run(q+1, feed_dict={q:[[1],[2],[3]]}))   # array([[2],[3],[4]])
    print(sess.run(p*r, feed_dict={p:3, r:np.eye(2)}))  # array([[3,0],[0,3]])

###############################################################################
# Fun with Variables (a dynamic wrapper around static Tensors)
###############################################################################

tf.reset_default_graph()
u = tf.Variable(3)             # This sets the value of the Tensor
v = tf.get_variable('v', (1))  # This sets the shape of the Tensor
w = tf.get_variable('w', (1, 2), initializer=tf.zeros_initializer)
x = tf.get_variable('x', dtype=tf.int32, initializer=tf.constant([23, 42]))

# Initializing Variables
with tf.Session() as sess:
    # ...with Variable.intializer
    try: sess.run(u)
    except Exception as e: print(e)  # 'Attempting to use uninitialized value Variable'
    sess.run(u.initializer)          # Variables must be initialized before running in a Session
    print(u)                         # Variable(name='Variable:0')
    print(sess.run(u))               # 3

    # ...with Variable.assign()
    try: sess.run(v)
    except Exception as e: print(e)  # 'Attempting to use uninitialized value v'
    sess.run(v.assign([100]))        # v.initializer is v.assign(v.initialized_value())
    print(v)                         # Variable(name='v:0')
    print(sess.run(v))               # array([100.])

    # ...with tf.global_variables_initializer()
    try: sess.run(w)
    except Exception as e: print(e)                       # 'Attempting to use uninitialized value v'
    print(sess.run(tf.report_uninitialized_variables()))  # ['w', 'x']
    sess.run(tf.global_variables_initializer())           # Now, all Variables can be used in the Session
    print(sess.run(w))                                    # array([[0., 0.]])
    print(sess.run(x+1))                                  # array([23, 42])


# After Variables are initialized, you can use them like Tensors in Sessions
with tf.Session() as sess:
    # Running a Session works the same way with Variables
    sess.run(tf.global_variables_initializer())
    print(sess.run(u))     # 3
    print(sess.run(10*u))  # 30

    # The Operation created by assign() is powerful as it allows us to modify a
    # Tensor after creation, in the middle of a Session run
    print(sess.run(u.assign(5)))        # 5
    print(sess.run(tf.assign(v, [1])))  # array([1.])

    # The static value of the Tensor is truly modified, as it is retained
    # between runs of a Session
    print(sess.run(u))  # 5
    print(sess.run(v))  # array([1.])

# When declaring Variables, if a name is specified, it must be unique
y = tf.get_variable('y', initializer=4)
try: z = tf.get_variable('y', initializer=2)
except Exception as e: print(e)  # 'Variable y already exists, disallowed'

# This makes it problematic to reuse Variable creation functions, so there is
# the concept of scope to namespace Variables
with tf.variable_scope('scopeA'):
    z1 = tf.get_variable('z', initializer=4)
with tf.variable_scope('scopeB'):
    z2 = tf.get_variable('z', initializer=4)
    z3 = tf.constant(3)
print(z1)                # Variable(name='scopeA/z:0')
print(z2)                # Variable(name='scopeB/z:0')
print(z3)                # Tensor('scopeB/Const_1:0')

# However, you can explicitly share scopes to duplicate references to Variables
with tf.variable_scope('scopeC'):
    t1 = tf.get_variable('t', dtype=tf.int64, initializer=np.array([1,1]))
with tf.variable_scope('scopeC', reuse=True):
    t2 = tf.get_variable('t', dtype=tf.int64)  # Parameters of reused Variables must be identical
print(t1)  # Variable(name='scopeC/t:0')
print(t2)  # Variable(name='scopeC/t:0')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(t1))                # array([1,1])
    print(sess.run(t2))                # array([1,1])
    print(sess.run(t1.assign([2,2])))  # array([2,2])
    print(sess.run(t2))                # array([2,2])

###############################################################################
# Fun with Graphs (Operations, Tensors, and Variables linked in a dataflow graph)
###############################################################################

# TensorFlow always maintains a default graph in the global scope
tf.reset_default_graph()
graph = tf.get_default_graph()
print(graph)  # A Graph object with no name

# All Tensors are added to the default graph
#  - Constants are stored in the Graph as the output of a Const Operation
#  - Placeholders are stored in the Graph as the output of a Placeholder Operation
a = tf.constant([1, 2])
p = tf.placeholder(tf.int32, shape=())
print(a is graph.get_tensor_by_name('Const:0'))            # True
print(p is graph.get_tensor_by_name('Placeholder:0'))      # True
print(repr(a.op))                                          # Operation(name='Const', type='Const')
print(repr(p.op))                                          # Operation(name='Placeholder', type='Placeholder')
print(a.op is graph.get_operation_by_name('Const'))        # True
print(p.op is graph.get_operation_by_name('Placeholder'))  # True

# A Tensor is the output of an Operation, and is named after that Operation
#  - Operations can have multiple input and output Tensors
#  - Tensors have only one source
print(a.op.inputs._inputs)                          # []
print(a.op.outputs)                                 # [Tensor('Const:0')]
print(p.op.inputs._inputs)                          # []
print(p.op.outputs)                                 # [Tensor('Placeholder:0')]
print(graph.get_operations())                       # A list of Operations

# An example of a multiply Operation with multiple input Tensors
result = p*a
print(result)                          # Tensor('mul:0')
print(repr(result.op))                 # Operation(name='mul', type='Mul')
print(result.op.inputs._inputs)        # [Tensor('Placeholder:0'), Tensor('Const:0')]
print(result.op.outputs)               # [Tensor('mul:0')]
print(result.op.outputs[0] is result)  # True

# You can give names to Operations by using tf.* functions
result = tf.multiply(p, a, name='result')
print(result)             # Tensor('result:0')
print(repr(result.op))    # Operation(name='result', type='Mul')
print(result.op.outputs)  # [Tensor('result:0')]

# Variables are stored in the Graph as the output of a VariableV2 Operation
v = tf.get_variable('v', initializer=1)
print(v)                    # Variable(name='v:0')
print(repr(v.op))           # Operation(name='v', type='VariableV2')
print(v.op.inputs._inputs)  # []
print(v.op.outputs)         # [Tensor('v:0')]

# The VariableV2 Operation has sub-Operations such as read and assign, which
# are (mostly transparently) used in the Graph to operate on Variables
# More info: https://stackoverflow.com/a/42798537/908744
v2 = tf.identity(v, name='v2')
print(v2)                    # Tensor('v2:0')
print(repr(v2.op))           # Operation(name='v2', type='Identity')
print(v2.op.inputs._inputs)  # [Tensor('v/read:0')]
print(v2.op.outputs)         # [Tensor('v2:0')]

# Graphs have collections that can hold groups of (usually) Variables
u = tf.get_variable('u', initializer=2, collections=['my_collection'])
tf.add_to_collection('my_collection', [{'arbitrary object'}])
print(tf.GraphKeys.GLOBAL_VARIABLES)                     # By default, all Variables go into this collection
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))  # [Variable(name='v:0')]
print(tf.get_collection('my_collection'))                # [Variable(name='u:0'), [{'arbitrary object'}]]

# You can manage multiple Graphs in the same script, each owning its own context
graphB = tf.Graph()
with graphB.as_default():
    c = tf.constant('Node in graphB')
    print(c.graph is graph)   # False
    print(c.graph is graphB)  # True

    with tf.Session() as sess:
        print(sess.run(c))  # 'Node in graphB'
