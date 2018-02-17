import os
import numpy as np
from collections import Counter

class Dataset():

    def __init__(self, batch_size=32, sequence_length=30):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.current_batch_index = 0

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters
        cntr = Counter(data)
        self.sorted_chars = sorted(cntr.keys(), key=cntr.get, reverse=True)

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
        # reverse the mapping
        self.id2char = {k:v for v,k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return np.array(list(map(self.id2char.get, encoded_sequence)))
    
    def decode_str(self, encoded_sequence):
        # returns the sequence decoded as string
        return "".join(self.decode(encoded_sequence))

    def create_minibatches(self):
        # shift data
        shifted_x = self.x[:-1]
        shifted_y = self.x[1:]
        # batch shape and total size
        batch_shape = (self.batch_size, self.sequence_length)
        step = self.batch_size * self.sequence_length

        self.num_batches = int(len(shifted_x) / step) 
        self.batches = []

        for batch_index in range(self.num_batches):
            batch_start_pos = batch_index * step
            batch_end_pos = batch_start_pos + step

            batch_x = np.array(shifted_x[batch_start_pos:batch_end_pos]).reshape(batch_shape)
            batch_y = np.array(shifted_y[batch_start_pos:batch_end_pos]).reshape(batch_shape)
            self.batches.append((batch_x, batch_y))

    def minibatch_generator(self):
        for epoch, (batch_x, batch_y) in enumerate(self.batches):
            new_epoch = epoch == 0
            yield new_epoch, batch_x, batch_y
            
    def next_minibatch(self):
        new_epoch = self.current_batch_index == 0
        batch_x, batch_y = self.batches[self.current_batch_index]
        self.current_batch_index += 1

        if self.current_batch_index >= self.num_batches:
            self.current_batch_index = 0

        return new_epoch, batch_x, batch_y

if __name__ == "__main__":
    ds = Dataset()
    ds.preprocess("data/selected_conversations.txt")
    print(ds.encode("Ana voli Milovana!"))
    print(ds.decode_str([26, 6, 5, 0, 36, 3, 12, 7, 0, 40, 7, 12, 3, 36, 5, 6, 5, 49]))
    ds.create_minibatches()
    for new, batch_x, batch_y in ds.minibatch_generator():
        new1, btx1, bty1 = ds.next_minibatch()
        print(new, new1)