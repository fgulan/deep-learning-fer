import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. definicija računskog grafa
# podatci i parametri
X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak
loss = (Y - Y_)**2

loss_grad_a = tf.reduce_sum(2 * (a * X + b - Y_) * X)
loss_grad_b = tf.reduce_sum(2 * (a * X + b - Y_))

# optimizacijski postupak: gradijentni
trainer = tf.train.GradientDescentOptimizer(0.1)
grads_and_vars = trainer.compute_gradients(loss, [a, b])
train_op = trainer.apply_gradients(grads_and_vars)
grads = [pair[0] for pair in grads_and_vars]
grads = tf.Print(grads, [grads], 'Gradients: ')

# 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3. učenje
# neka igre počnu!
for i in range(100):
    val_loss, _, val_a, val_b, val_grads, val_grad_a, val_grad_b = sess.run(
        [loss, train_op, a, b, grads, loss_grad_a, loss_grad_b],
        feed_dict={X: [1, 2], Y_: [3, 5]})
    print(i, val_loss, val_a, val_b, val_grads, [val_grad_a, val_grad_b])
