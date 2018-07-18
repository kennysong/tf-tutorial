import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

# Tensors are immediately evaluated and have values
a = tf.constant([[1, 2]])
b = tf.matmul(a, tf.transpose(a))
print(a)          # Tensor([[1,2]])
print(a.numpy())  # array([[1,2]])
print(b)          # Tensor([[5]])

# EagerTensors can also be used in control flows
c = tf.constant(3)
d = tfe.Variable(4)
if c > 0 and int(d % 2) == 0 and tf.equal(d, 4):
    print('yes')  # 'yes'

# Defining a function and creating its gradient function
def f(x, y): return x**2 - x*y
grad_f = tfe.gradients_function(f)
print(f(2., 1.))       # 2.0
print(grad_f(2., 1.))  # [Tensor(3.0), Tensor(-2.0)]

# It's also possible to calculate partial derivatives wrt. Variables implicitly
# involved in the computation of the function
x = tfe.Variable(1.0, name='x')
y = tfe.Variable(2.0, name='y')
def g(z): return z*x + z*y**2
grad_g = tfe.gradients_function(g)
imp_grad_g = tfe.implicit_gradients(g)

print(g(3.))           # Tensor(15.0)
print(grad_g(3.))      # [Tensor(5.0)] is [dg/dz]
print(imp_grad_g(3.))  # [(Tensor(3.0), Variable(name='x:0')),
                       #  (Tensor(12.0), Variable(name='y:0'))] is [dg/dx, dg/dy]

# You can also calculate partial derivatives of one computation inside a
# GradientTape context manager
x = tf.constant(3.)
y = tfe.Variable(3.)
with tfe.GradientTape() as tape:
    tape.watch(x)  # Trainable Variables are automatically watched
    f = x**2 + y**2
print(tape.gradient(f, [x, y]))
