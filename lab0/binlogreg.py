import data
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(prob, y):
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)

def binlogreg_train(X, Y_, param_niter=10000, param_delta=0.8):
    '''
    Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
        w, b: parametri logističke regresije
    '''
    b = 0
    w = np.random.randn(2)
    N = len(X)

    for i in range(param_niter):
        scores = np.dot(X, w) + b 
        probs = sigmoid(scores)
        loss = np.sum(cross_entropy_loss(probs, Y_)) 
        
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_   
        
        grad_w = 1.0/N  * np.dot(dL_dscores, X)
        grad_b = 1.0/N  * np.sum(dL_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b
    
    return w, b

def binlogreg_classify(X, w, b):
    '''
    Argumenti
        X:    podatci, np.array NxD
        w, b: parametri logističke regresije 

    Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''
    return sigmoid(np.dot(X, w) + b)

def binlogreg_decfun(w,b):
    def classify(X):
        return binlogreg_classify(X, w,b)
    return classify

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y = np.array(probs > 0.5, dtype=np.int32)

    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)