import collections
import os
import string

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

from datasets import preprocess


def read_chars(filename):
    with tf.gfile.GFile(filename, "r") as f:
        chars = list()
        messages = f.read()\
            .replace("\r\n", "\n")\
            .replace(preprocess.message_delimiter + "\n", preprocess.message_delimiter)\
            .split(preprocess.message_delimiter)

        # Throw out messages containing anything other than basic alphanumeric, punctuation, and linefeed chars
        for message in messages:
            for c in message:
                if not (c.isdigit()
                        or ord("a") <= ord(c) <= ord("z")
                        or ord("A") <= ord(c) <= ord("Z")
                        or c in string.punctuation
                        or c == " "
                        or c == "\n"
                        or c == "\r"):
                    break
            else:
                chars.extend(list(message))
                chars.append(preprocess.message_delimiter)

        # Splits the string read from the file into a list of the individual characters
        return chars


def build_alphabet(filename):
    data = [c for c in read_chars(filename)]
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    chars, _ = list(zip(*count_pairs))
    char_to_id = dict(zip(chars, range(len(chars))))

    return char_to_id


def file_to_char_ids(filename, char_to_id):
    data = read_chars(filename)
    return [char_to_id[char] for char in data if char in char_to_id]


def load_data(data_path):
    train_path = os.path.join(data_path, "train.txt")
    test_path = os.path.join(data_path, "test.txt")

    char_to_id = build_alphabet(train_path)
    train_data = file_to_char_ids(train_path, char_to_id)
    test_data = file_to_char_ids(test_path, char_to_id)

    charset_size = len(char_to_id)
    reversed_dictionary = dict(zip(char_to_id.values(), char_to_id.keys()))

    print(train_data[:5])
    print(char_to_id)
    print(charset_size)
    print("".join([reversed_dictionary[x] for x in train_data[:30]]))
    print(" ".join([str(x) for x in train_data[:30]]))

    return train_data, test_data, charset_size, reversed_dictionary


# train_data, test_data, charset_size, reversed_dictionary =\
#     load_data(os.path.abspath(r"..\datasets\krishamomo4055\trash"))
# with open(r"..\models\trainer_1" + r"\reversed_dictionary.pkl", "wb") as f:
#     reversed_dictionary["message_delimiter"] = preprocess.message_delimiter
#     pickle.dump(reversed_dictionary, f, pickle.HIGHEST_PROTOCOL)


class KerasBatchGenerator(object):

    def __init__(self, data, block_size, batch_size, charset_size, skip_size=5):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.charset_size = charset_size
        self.skip_size = skip_size

        self.current_idx = 0

    def generate(self):
        x = np.zeros((self.batch_size, self.block_size))
        y = np.zeros((self.batch_size, self.block_size, self.charset_size))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.block_size >= len(self.data):
                    self.current_idx = 0

                x[i, :] = self.data[self.current_idx:self.current_idx + self.block_size]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.block_size + 1]

                y[i, :, :] = to_categorical(temp_y, num_classes=self.charset_size)
                self.current_idx += self.skip_size
            yield x, y
