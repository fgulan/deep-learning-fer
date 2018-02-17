import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFDeep:
    def __init__(self, layers, param_delta=0.1, param_lambda=1e-4, 
                 activation=tf.nn.relu, optimizer=tf.train.GradientDescentOptimizer, 
                 use_decay=False):
        """Arguments:
            - layer: list with number of neurons in each layer
            - param_delta: training step
            - param_labmda: regularization
            - activation: activation function in hidden layers
        """
        layers_count = len(layers)
        if layers_count < 2:
            raise ValueError("Deep object must have at least two layers!")
        # definicija podataka i parametara:
        self.X = tf.placeholder(tf.float32, [None, layers[0]])
        self.Yoh_ = tf.placeholder(tf.float32, [None, layers[-1]])

        self.Ws, self.bs = [], []
        self.probs = self.X
        for i in range(1, layers_count):
            self.Ws.append(tf.Variable(tf.random_normal([layers[i-1], layers[i]]),
                                                        name="W_{0}{1}".format(i - 1, i)))
            self.bs.append(tf.Variable(tf.random_normal([layers[i]]),
                                                        name="b_{0}".format(i)))

        for i in range(len(self.Ws) - 1):
            self.probs = activation(tf.matmul(self.probs, self.Ws[i]) + self.bs[i])
        self.probs = tf.nn.softmax(tf.matmul(self.probs, self.Ws[-1]) + self.bs[-1])
        
        # formulacija gubitka: self.loss
        log_probs = tf.log(tf.clip_by_value(self.probs, 1e-10, 1.0)) # hackity hack for nan issue
        err_loss = -tf.reduce_sum(self.Yoh_ * log_probs, 1)
        reg_loss = sum(map(lambda w : tf.nn.l2_loss(w), self.Ws))
        self.loss = tf.reduce_mean(err_loss) + param_lambda * reg_loss
    
        global_step = None
        if use_decay:
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = param_delta
            param_delta = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                     decay_steps=1, decay_rate=1-1e-4, staircase=True)
        # formulacija operacije učenja: self.train_step
        self.train_step = optimizer(param_delta).minimize(self.loss, global_step=global_step)
    
        # instanciranje izvedbenog konteksta: self.session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
    
    def train(self, X, Yoh_, param_niter, print_step=1000):
        """Arguments:
            - X: actual datapoints [NxD]
            - Yoh_: one-hot encoded labels [NxC]
            - param_niter: number of iterations
        """
        # incijalizacija parametara
        feed_dict = {self.X: X, self.Yoh_: Yoh_}

        # optimizacijska petlja
        for i in range(param_niter):
            train_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            if i % print_step == 0: 
                print("Iter {} => loss: {}".format(i, train_loss))
    
    def train_val(self, X, Yoh_, X_val, Y_val_oh_, param_niter, validation_step=10):  
        best_loss = -1
        best_Ws, best_bs = self.Ws, self.bs

        for i in range(param_niter):
            train_loss, _ = self.session.run([self.loss, self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % validation_step == 0: 
                val_loss = self.session.run([self.loss], feed_dict={self.X: X_val, self.Yoh_: Y_val_oh_})[0]
                print("Iter {0} => val loss: {1}, train loss: {2}".format(i, val_loss, train_loss))
                if val_loss < best_loss or best_loss == -1:
                    print("Saving model")
                    best_loss = val_loss
                    best_Ws = self.Ws
                    best_bs = self.bs
        
        self.Ws, self.bs = best_Ws, best_bs
        
    def train_mb(self, X, Yoh_, param_niter, batch_size=32, print_step=10):
        train_loss = -1
        for i in range(param_niter):
            for batch in self.iterate_minibatches(X, Yoh_, batch_size):
                X_batch, Yoh_batch = batch
                train_loss, _ = self.session.run([self.loss, self.train_step], 
                                                 feed_dict={self.X: X_batch, self.Yoh_: Yoh_batch})
            if i % print_step == 0: 
                print("Iter {} => loss: {}".format(i, train_loss))

    def eval(self, X):
        """Arguments:
            - X: actual datapoints [NxD]
            Returns: predicted class probabilites [NxC]
        """
        probs = self.session.run(self.probs, {self.X: X})
        return probs

    def count_params(self):
        total_count = 0
        for var in tf.trainable_variables():
            print(var.name, var.shape)
            if len(var.shape) == 2:
                total_count += var.shape[0].value * var.shape[1].value
            else:
                total_count += var.shape[0].value
        print("Total count: ", str(total_count))

    def iterate_minibatches(self, X, Y_, batch_size):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
            indices_range = indices[start_idx:start_idx + batch_size]
            yield X[indices_range], Y_[indices_range]

    def get_Ws(self):
        return self.session.run(self.Ws)

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm(6, 2, 10)
    # X, Y_ = data.sample_gauss(3, 100)
    Yoh_ = data.class_to_onehot(Y_)

    layers = [X.shape[1], 10, 10, Yoh_.shape[1]]
    model = TFDeep(layers, activation=tf.nn.relu)
    model.count_params()
    model.train(X, Yoh_, 10000)
    model.count_params()

    # dohvati vjerojatnosti na skupu za učenje
    Y = model.eval(X).argmax(axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    print(data.eval_perf_multi(Y, Y_))

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda x: model.eval(x)[:,0]
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()