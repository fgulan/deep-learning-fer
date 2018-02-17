import data
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs

def logreg_train(X, Y_, param_niter=10000, param_delta=0.8):
    '''
    Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
        w, b: parametri logističke regresije
    '''
    C = np.max(Y_) + 1
    n_samples, n_features = X.shape
    W = np.random.randn(C, n_features)
    b = np.zeros(C)

    for i in range(param_niter):
        scores = np.dot(X, W.T) + b
        expscores = np.exp(scores - np.max(scores))
        sumexp = expscores.sum(axis=1)
        probs = expscores / sumexp.reshape(-1,1)
        logprobs = -np.log(probs[range(len(X)), Y_])   # N*1 
        loss  = logprobs.sum()
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs   # N x C
        dL_ds[range(len(X)),Y_] -= 1

        # gradijenti parametara
        grad_W = 1.0/n_samples  * np.dot(dL_ds.T, X)
        grad_b = 1.0/n_samples  * dL_ds.sum(axis=0)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b
    return W, b

def logreg_classify(X, W, b):
    '''
    Argumenti
        X:    podatci, np.array NxD
        w, b: parametri logističke regresije 

    Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''
    scores = np.dot(X, W.T) + b    # N x C
    expscores = np.exp(scores - np.max(scores))    # N x C

    # nazivnik sofmaksa
    sumexp = expscores.sum(axis=1)    # N x 1
    return expscores / sumexp.reshape(-1,1)

def logreg_decfun(X, W, b):
    def classify(X):
        return logreg_classify(X, W, b).argmax(axis=1)
    return classify

if __name__=="__main__":
    np.random.seed(100)
    X, Y_ = data.sample_gauss(3, 100)

    # train the logistic regression model
    W, b = logreg_train(X, Y_, param_niter=100000, param_delta=0.7)

    # evaluate the model on the train set
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    # graph the decision surface
    decfun = logreg_decfun(X, W, b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)

    plt.show()
