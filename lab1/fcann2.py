import data
import numpy as np
import matplotlib.pyplot as plt
    
def fcann2_train(X, Y_, param_niter=100000, param_delta=0.05, param_lambda=1e-3, param_hidden_dim=5, print_step=1000):
    N, input_dim = X.shape
    C = np.max(Y_) + 1
    Y_oh = data.class_to_onehot(Y_)

    W_1 = 0.01 * np.random.randn(input_dim, param_hidden_dim)
    b_1 = np.zeros((1, param_hidden_dim))
    W_2 = 0.01 * np.random.randn(param_hidden_dim, C)
    b_2 = np.zeros((1, C))

    for i in range(param_niter):
        probs, hidden_layer = forward_pass(X, W_1, b_1, W_2, b_2)
        logprobs = -np.log(probs[range(N), Y_])
        data_loss = np.sum(logprobs) / N
        reg_loss = 0.5 * param_lambda * (np.sum(W_1*W_1) + np.sum(W_2*W_2))
        loss = data_loss + reg_loss
        
        if i % print_step == 0:
            print("iteration {}: loss {}".format(i, loss))
        
        dscores = probs - Y_oh
        dscores /= N

        grad_W2 = np.dot(hidden_layer.T, dscores)
        grad_b2 = np.sum(dscores, axis=0, keepdims=True)

        grad_hidden = np.dot(dscores, W_2.T)
        grad_hidden[hidden_layer <= 0] = 0

        grad_W1 = np.dot(X.T, grad_hidden)
        grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)

        grad_W2 += param_lambda * W_2
        grad_W1 += param_lambda * W_1

        W_1 -= param_delta * grad_W1
        b_1 -= param_delta * grad_b1
        W_2 -= param_delta * grad_W2
        b_2 -= param_delta * grad_b2
    
    return W_1, b_1, W_2, b_2

def forward_pass(X, W_1, b_1, W_2, b_2):
    hidden_layer = np.maximum(0, np.dot(X, W_1) + b_1) # ReLU 
    scores = np.dot(hidden_layer, W_2) + b_2
    exp_scores = np.exp(scores - np.max(scores))
    return (exp_scores / np.sum(exp_scores, axis=1, keepdims=True), hidden_layer)
    
def fcann2_classify(X, W_1, b_1, W_2, b_2):
    probs, _ = forward_pass(X, W_1, b_1, W_2, b_2)
    return probs

def fcann2_classify_new(W_1, b_1, W_2, b_2):
    def classify(X):
        return fcann2_classify(X, W_1, b_1, W_2, b_2)[:,0]
    return classify

if __name__=="__main__":
    np.random.seed(100)
    X, Y_ = data.sample_gmm(6, 2, 10)
    W1, b1, W2, b2 = fcann2_train(X, Y_)
    decfunc = fcann2_classify_new(W1, b1, W2, b2)

    Y = np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfunc, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)

    plt.show()
