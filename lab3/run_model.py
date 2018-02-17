from dataset import Dataset
from rnn import SimpleRNN
import numpy as np

def one_hot(batch, vocab_size):

    def _oh(x, vocab_size):
        x_oh = np.zeros((x.shape[0], vocab_size))
        x_oh[np.arange(x.shape[0]), x] = 1
        return x_oh

    if batch.ndim == 1:
        return _oh(batch, vocab_size)
    else:
        return np.array([_oh(s, vocab_size) for s in batch])

def sample(seed, n_sample, rnn):
    h0 = np.zeros((1, rnn.hidden_size))
    seed_onehot = one_hot(seed, rnn.vocab_size)

    h = h0
    for char in seed_onehot:
        h, _ = rnn.rnn_step_forward(char[np.newaxis, :], h)

    sample = np.zeros((n_sample, ), dtype=np.int32)
    sample[:len(seed)] = seed
    for i in range(len(seed), n_sample):
        # Calculate probabilistic output
        model_out = rnn.output(h[np.newaxis, :, :])
        # Choose next letter in sample with defined probabilty
        sample[i] = np.random.choice(np.arange(model_out.shape[-1]), p=model_out.ravel())
        # Forward current letter
        next_input = np.zeros((1, rnn.vocab_size))
        next_input[0, sample[i]] = 1
        h, _ = rnn.rnn_step_forward(next_input, h)

    return sample

def run_language_model(dataset, max_epochs, seed, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    vocab_size = len(dataset.sorted_chars)
    RNN = SimpleRNN(hidden_size, sequence_length, vocab_size, learning_rate)

    current_epoch = 0 
    batch = 0

    h0 = np.zeros((dataset.batch_size, hidden_size))

    average_loss = 0

    while current_epoch < max_epochs: 
        next_epoch, x, y = dataset.next_minibatch()
        
        if next_epoch:
            batch = 0
            current_epoch += 1
            h0 = np.zeros_like(h0)

        x_oh, y_oh = one_hot(x, vocab_size), one_hot(y, vocab_size)
        loss, h0 = RNN.step(h0, x_oh, y_oh)
        average_loss = 0.9 * average_loss + 0.1 * loss

        if batch % sample_every == 0: 
            print("==== batch: {0}/{1}, bch loss: {2}, epoch: {3}/{4}, avg loss: {5} ===="
                    .format(batch, dataset.num_batches, loss, current_epoch, max_epochs, average_loss))
            sample_encoded = sample(dataset.encode(seed), 200, RNN)
            print("===============================================================================================")
            print(dataset.decode_str(sample_encoded))
            print("===============================================================================================")

        batch += 1

if __name__ == "__main__":
    # Prepare dataset
    dataset = Dataset(sequence_length=30, batch_size=32)
    dataset.preprocess("data/selected_conversations.txt")
    dataset.create_minibatches()

    # Start learning
    run_language_model(dataset, 50, sequence_length=dataset.sequence_length, seed="HAN:\nIs that good or bad?\n\n")