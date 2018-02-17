import tensorflow as tf
import numpy as np
from tf_deep import TFDeep
from data import eval_perf_multi
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

tf.set_random_seed(100)
np.random.seed(100)

N = mnist.train.images.shape[0]
D = mnist.train.images.shape[1]
C = mnist.train.labels.shape[1]
print("Loaded data")

Y_test_ = np.argmax(mnist.test.labels, axis=1)
model = TFDeep([D, C], param_delta=0.15, param_lambda=0.001)
model.train(mnist.train.images, mnist.train.labels, 500, print_step=50)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
eval_perf_multi(Y_test_, Y_test)

Y_train_ = np.argmax(mnist.train.labels, axis=1)
Y_test_ = np.argmax(mnist.test.labels, axis=1)
model1 = TFDeep([D, 100, C], param_delta=0.0015, param_lambda=0.000001)
model1.train(mnist.train.images, mnist.train.labels, 250, print_step=50)
Y_test = np.argmax(model1.eval(mnist.test.images), axis=1)
eval_perf_multi(Y_test_, Y_test)

X_train, X_valid, y_train, y_valid = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.2, random_state=42)
model = TFDeep([D, C], param_delta=0.15, param_lambda=0.001)
model.train_val(X_train, y_train, X_valid, y_valid, 500)
Y_test_ = np.argmax(mnist.test.labels, axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
eval_perf_multi(Y_test_, Y_test)

model = TFDeep([D, C], param_delta=0.15, param_lambda=0.01)
model.train_mb(mnist.train.images, mnist.train.labels, 500, 100)
Y_test_ = np.argmax(mnist.test.labels, axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
eval_perf_multi(Y_test_, Y_test)

W = model.get_Ws()[0]
for i in range(10):
    plt.imshow(W[:,i].reshape(28,28), cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()

model = TFDeep([D, C], param_delta=1e-4, param_lambda=0.01, optimizer=tf.train.AdamOptimizer)
model.train_mb(mnist.train.images, mnist.train.labels, 200, 100)
Y_test_ = np.argmax(mnist.test.labels, axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
eval_perf_multi(Y_test_, Y_test)

model = TFDeep([D, 100, C], param_delta=1e-3, param_lambda=0.001, optimizer=tf.train.AdamOptimizer, use_decay=True)
model.train_mb(mnist.train.images, mnist.train.labels, 200, 100)
Y_test_ = np.argmax(mnist.test.labels, axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
eval_perf_multi(Y_test_, Y_test)


model = svm.SVC(decision_function_shape='ovo', cache_size=1000)
model.fit(mnist.train.images, np.argmax(mnist.train.labels, axis=1))
Y_test_ = np.argmax(mnist.test.labels, axis=1)
Y_test = model.predict(mnist.test.images)
eval_perf_multi(Y_test_, Y_test)

model = svm.SVC(decision_function_shape='ovo', kernel='linear', cache_size=1000)
model.fit(mnist.train.images, np.argmax(mnist.train.labels, axis=1))
Y_test_ = np.argmax(mnist.test.labels, axis=1)
Y_test = model.predict(mnist.test.images)
eval_perf_multi(Y_test_, Y_test)