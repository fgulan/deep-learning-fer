import numpy as np
from sklearn.metrics import log_loss

def softmax(value):
    max_value = np.max(value, axis=2)
    max_value = max_value[:, :, np.newaxis] # Broadcast it manually
    exp = np.exp(value - max_value)
    return exp / np.sum(exp, axis=2, keepdims=True)

class SimpleRNN:

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        hidden_xavier_scale = 1.0 / np.sqrt(hidden_size)
        self.U = np.random.normal(size=[vocab_size, hidden_size], scale=hidden_xavier_scale) # ... input projection
        self.W = np.random.normal(size=[hidden_size, hidden_size], scale=hidden_xavier_scale) # ... hidden-to-hidden projection
        self.b = np.zeros([1, hidden_size]) # ... input bias

        self.V = np.random.normal(size=[hidden_size, vocab_size], scale=1.0 / np.sqrt(vocab_size)) # ... output projection
        self.c = np.zeros([1, vocab_size]) # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U=None, W=None, b=None):
        # A single time step forward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        
        # Extract values if given
        U = self.U if U is None else U
        W = self.W if W is None else W
        b = self.b if b is None else b

        h_current = np.tanh(np.dot(h_prev, W) + np.dot(x, U) + b)
        cache = (h_current, h_prev, x)
        return h_current, cache

    def rnn_forward(self, x, h0, U=None, W=None, b=None):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        
        # Extract values if given
        U = self.U if U is None else U
        W = self.W if W is None else W
        b = self.b if b is None else b

        h, cache = [h0], []
        sequences = x.transpose(1, 0, 2)
        
        for sequence in sequences:
            h_current, cache_current = self.rnn_step_forward(sequence, h[-1], U, W, b)
            cache.append(cache_current)
            h.append(h_current)
        
        # Skip initial hidden state
        h = np.array(h[1:]).transpose(1, 0, 2)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        return h, cache

    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
        h_current, h_prev, x = cache

        # compute and return gradients with respect to each parameter
        da = grad_next * (1 - h_current**2)
        dh_prev = np.dot(da, self.W.T)
        dU = np.dot(x.T, da) / grad_next.shape[0]
        dW = np.dot(h_prev.T, da) / grad_next.shape[0]
        db = np.sum(da, axis=0) / grad_next.shape[0]

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        dU, dW, db = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.b)

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
        dh = dh.transpose(1, 0, 2)

        dh_prev = np.zeros_like(dh[-1])
        for dh_current, cache_current in reversed(list(zip(dh, cache))):
            dh_prev, dU_current, dW_current, db_current = self.rnn_step_backward(dh_current + dh_prev, cache_current)
            dU += dU_current
            dW += dW_current
            db += db_current

        return dU, dW, db

    def output(self, h, V=None, c=None):
        # Calculate the output probabilities of the network
        # Extract values if given
        V = self.V if V is None else V
        c = self.c if c is None else c
        logits = np.dot(h, V) + c
        return softmax(logits)

    def output_loss_and_grads(self, h, y, V=None, c=None):
        # Calculate the loss of the network for each of the outputs
        
        # h - hidden states of the network for each timestep. 
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension batch_size x sequence_length x vocabulary size

        # Extract values if given
        V = self.V if V is None else V
        c = self.c if c is None else c

        batch_size = h.shape[0]
        y_out = self.output(h, V, c)

        log_loss_ = log_loss(y.reshape(-1, self.vocab_size), y_out.reshape(-1, self.vocab_size))
        loss = log_loss_ * self.sequence_length  # Since it computes average cross_entropy loss
        d_out = y_out - y
        
        dh, dV, dc = [], np.zeros_like(V), np.zeros_like(c)
        
        for d_out_current, h_current in zip(d_out.transpose(1, 0, 2), h.transpose(1, 0, 2)):
            dV += np.dot(h_current.T, d_out_current) / batch_size
            dc += np.average(d_out_current, axis=0)
            dh.append(np.dot(d_out_current, V.T))

        dh = np.array(dh).transpose(1, 0, 2)

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc, eps=1e-6):
        # update memory matrices
        self.memory_U += np.square(dU)
        self.memory_W += np.square(dW)
        self.memory_b += np.square(db)
        self.memory_V += np.square(dV)
        self.memory_c += np.square(dc)
        
        # perform the Adagrad update of parameters
        self.U -= self.learning_rate * dU / np.sqrt(self.memory_U + eps)
        self.W -= self.learning_rate * dW / np.sqrt(self.memory_W + eps)
        self.b -= self.learning_rate * db / np.sqrt(self.memory_b + eps)
        self.V -= self.learning_rate * dV / np.sqrt(self.memory_V + eps)
        self.c -= self.learning_rate * dc / np.sqrt(self.memory_c + eps)

    def step(self, h0, x_oh, y_oh):
        h, cache = self.rnn_forward(x_oh, h0)
        loss, dh, dV, dc = self.output_loss_and_grads(h, y_oh)
        dU, dW, db = self.rnn_backward(dh, cache)

        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)
        dV = np.clip(dV, -5, 5)
        dc = np.clip(dc, -5, 5)

        self.update(dU, dW, db, dV, dc)
        return loss, h[:, -1, :]