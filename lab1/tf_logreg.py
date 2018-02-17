import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFLogreg:
    
    def __init__(self, D, C, param_delta=0.5, param_lambda=1e-3):
        """Arguments:
            - D: dimensions of each datapoint 
            - C: number of classes
            - param_delta: training step
            - param_labmda: regularization
        """
        # definicija podataka i parametara:
        self.X  = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        W = tf.Variable(tf.random_normal([C, D]))
        b = tf.Variable(tf.zeros([C]))

        # formulacija modela: izračunati self.probs
        scores = tf.matmul(self.X, W, transpose_b=True) + b
        self.probs = tf.nn.softmax(scores)

        # formulacija gubitka: self.loss
        log_probs = -tf.log(self.probs)
        err_loss = tf.reduce_sum(self.Yoh_ * log_probs, 1)
        reg_loss = tf.nn.l2_loss(W)
        self.loss = tf.reduce_mean(err_loss) + param_lambda * reg_loss
        
        # formulacija operacije učenja: self.train_step
        trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = trainer.minimize(self.loss)

        # instanciranje izvedbenog konteksta: self.session
        self.session = tf.Session()

    def train(self, X, Yoh_, param_niter, print_step=1000):
        """Arguments:
            - X: actual datapoints [NxD]
            - Yoh_: one-hot encoded labels [NxC]
            - param_niter: number of iterations
        """
        # incijalizacija parametara
        self.session.run(tf.global_variables_initializer())
        feed_dict = {self.X: X, self.Yoh_: Yoh_}

        # optimizacijska petlja
        for i in range(param_niter):
            val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            if i % print_step == 0: 
                print("Iter {} => loss: {}".format(i, val_loss))

    def eval(self, X):
        """Arguments:
            - X: actual datapoints [NxD]
            Returns: predicted class probabilites [NxC]
        """
        probs = self.session.run(self.probs, {self.X: X})
        return probs

if __name__ == "__main__":
      # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss(2, 100)
    Yoh_ = data.class_to_onehot(Y_)

    # izgradi graf:
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.5)

    # nauči parametre:
    tflr.train(X, Yoh_, 2000)

    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(X)
    Y = probs.argmax(axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print("Accuracy: {}".format(accuracy))
    print("Preciznost: \n{}".format(precision))
    print("Odziv: \n{}".format(recall))

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda x: tflr.eval(x).max(axis=1)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()