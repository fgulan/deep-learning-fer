import time
import os
import math
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.io
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './mnist/'
SAVE_DIR = "./out_l2_tf/"
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 8
BATCH_SIZE = 50
LR_POLICY = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def get_mnist_dataset(data_dir):    
    dataset = input_data.read_data_sets(data_dir, one_hot=True)
    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 1, 28, 28])
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 1, 28, 28])
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 1, 28, 28])
    test_y = dataset.test.labels
    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean
    import pdb; pdb.set_trace()
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def conv_layer(inputs, filters=32, kernel_size=[5, 5], 
               activation=tf.nn.relu, regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY), name=None):
    return tf.layers.conv2d(inputs, filters, kernel_size, padding='same', 
                            activation=activation, kernel_regularizer=regularizer,
                            kernel_initializer=tf.variance_scaling_initializer(), name=name)

def max_pool_layer(inputs, pool_size=[2, 2], strides=2, name=None):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding='same', name=name)

def fc_layer(input, units, activation=tf.nn.relu, regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY), name=None):
    return tf.layers.dense(input, units, activation, kernel_regularizer=regularizer, 
                           kernel_initializer=tf.variance_scaling_initializer(), name=name)

def build_dnn(inputs):
    input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
    conv1 = conv_layer(input_layer, 16, name="conv1")
    pool1 = max_pool_layer(conv1, name="pool1")

    conv2 = conv_layer(pool1, 32, name="conv2")
    pool2 = max_pool_layer(conv2, name="pool2")
    
    flat_pool2 = tf.contrib.layers.flatten(pool2)

    fc1 = fc_layer(flat_pool2, 512, name="fc1")
    logits = fc_layer(fc1, 10, activation=None, regularizer=None, name="logits")
    return logits

def iterate_minibatches(X, Y_, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        indices_range = indices[start_idx:start_idx + batch_size]
        yield X[indices_range], Y_[indices_range]

def get_trainable_weights():
    weights = []
    for var in tf.trainable_variables():
        if "/kernel:0" not in var.name: 
            continue
        weights.append(var)
    return weights

def draw_conv_filters(session, layer, epoch, step, name, save_dir):
    weights = session.run(layer).copy()
    num_filters = weights.shape[3]
    num_channels = weights.shape[2]
    k = weights.shape[0]
    assert weights.shape[0] == weights.shape[1]
    weights -= weights.min()
    weights /= weights.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = weights[:,:,:,i]
        
    img = img.reshape(height, width)
    filename = '%s_epoch_%02d_step_%06d.png' % (name, epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)

def fit():
    tf.reset_default_graph()
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_mnist_dataset(DATA_DIR)
    x = tf.placeholder(tf.float32, [None, 1, 28, 28])
    y_ = tf.placeholder(tf.float32, [None, 10])

    y_conv = build_dnn(x)
    weights = get_trainable_weights()[:-1]
    err_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    reg_loss = sum(map(lambda w : tf.nn.l2_loss(w), weights))
    loss = err_loss + WEIGHT_DECAY * reg_loss
    lr = tf.placeholder(tf.float32)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_examples = train_x.shape[0]
    assert num_examples % BATCH_SIZE == 0
    num_batches = num_examples // BATCH_SIZE
    print_step = 5
    draw_step = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(1, MAX_EPOCHS + 1):
            if epoch in LR_POLICY:
                lr_policy = LR_POLICY[epoch]['lr']
            for (j, batch) in enumerate(iterate_minibatches(train_x, train_y, BATCH_SIZE)):
                train_step.run(feed_dict={x: batch[0], y_: batch[1], lr: lr_policy})
                if j % print_step == 0:
                    batch_loss = loss.eval(feed_dict={x: batch[0], y_: batch[1]})
                    print('epoch %d, step %d/%d, batch loss %g' % (epoch, j + print_step, num_batches, batch_loss), end="\r", flush=True)
                if j % draw_step == 0:
                    draw_conv_filters(sess, weights[0], epoch, j, "conv1", SAVE_DIR)

            print("", flush=True)
            valid_acc = accuracy.eval(feed_dict={x: valid_x, y_: valid_y})
            valid_loss = loss.eval(feed_dict={x: valid_x, y_: valid_y})
            print('epoch %d, valid loss %g, valid accuracy %g' % (epoch, valid_loss, valid_acc))

        test_acc = accuracy.eval(feed_dict={x: test_x, y_: test_y})
        test_loss = loss.eval(feed_dict={x: test_x, y_: test_y})
        print('test loss %g, test accuracy %g' % (test_loss, test_acc))

fit()