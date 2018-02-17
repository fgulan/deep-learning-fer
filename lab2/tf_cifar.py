import time
import os
import math
import pickle
import skimage.io
import numpy as np
import tensorflow as tf
import skimage as ski

DATA_DIR = './cifar/'
SAVE_DIR = "./out_cifar/"
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 196
MAX_EPOCHS = 100
WEIGHT_DECAY = 1e-4

def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def to_one_hot(class_index, class_num=10):
    one_hot = [0] * class_num
    one_hot[class_index] = 1
    return one_hot

def load_cifar(data_dir):
    train_x = np.ndarray((0, IMG_HEIGHT * IMG_WIDTH * NUM_CHANNELS), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)).transpose(0,2,3,1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)).transpose(0,2,3,1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0,1,2))
    data_std = train_x.std((0,1,2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.reshape([-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
    valid_x = valid_x.reshape([-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
    test_x = test_x.reshape([-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
    train_y = np.array(list(map(to_one_hot, train_y)))
    valid_y = np.array(list(map(to_one_hot, valid_y)))
    test_y = np.array(list(map(to_one_hot, test_y)))
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def conv_layer(inputs, filters=32, kernel_size=[5, 5], 
               activation=tf.nn.relu, regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY), name=None):
    return tf.layers.conv2d(inputs, filters, kernel_size, padding='same', 
                            activation=activation, kernel_regularizer=regularizer, name=name)

def max_pool_layer(inputs, pool_size=[2, 2], strides=2, name=None):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding='same', name=name)

def fc_layer(input, units, activation=tf.nn.relu, regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY), name=None):
    return tf.layers.dense(input, units, activation, kernel_regularizer=regularizer, name=name)

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

def build_dnn(inputs):
    input_layer = tf.reshape(inputs, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
    conv1 = conv_layer(input_layer, 16, name="conv1")
    pool1 = max_pool_layer(conv1, name="pool1")

    conv2 = conv_layer(pool1, 32, name="conv2")
    pool2 = max_pool_layer(conv2, name="pool2")
    
    flat_pool2 = tf.contrib.layers.flatten(pool2)

    fc1 = fc_layer(flat_pool2, 256, name="fc1")
    fc2 = fc_layer(fc1, 128, name="fc2")
    logits = fc_layer(fc2, NUM_CLASSES, activation=None, regularizer=None, name="logits")
    return logits

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
    filename = '%s_epoch_%02d_step_%06d.png' % (name, epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)

def fit():
    tf.reset_default_graph()
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_cifar(DATA_DIR)
    x = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    y_conv = build_dnn(x)
    weights = get_trainable_weights()
    err_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    reg_loss = sum(map(lambda w : tf.nn.l2_loss(w), weights))
    loss = err_loss + WEIGHT_DECAY * reg_loss

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.01, global_step, 500, 0.95, staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_examples = train_x.shape[0]
    num_batches = num_examples // BATCH_SIZE
    print_step = 5
    draw_step = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(1, MAX_EPOCHS + 1):
            for (j, batch) in enumerate(iterate_minibatches(train_x, train_y, BATCH_SIZE)):
                train_step.run(feed_dict={x: batch[0], y_: batch[1]})
                if j % print_step == 0:
                    batch_loss = loss.eval(feed_dict={x: batch[0], y_: batch[1]})
                    print('epoch %d, step %d/%d, batch loss %g, learning rate: %g' % (epoch, j, num_batches, batch_loss, sess.run(learning_rate)), end="\r", flush=True)
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

