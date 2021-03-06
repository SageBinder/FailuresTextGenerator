import argparse
import os
import pickle

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from data_loader import load_data
from datasets import preprocess

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("model_dir", type=str)
parser.add_argument("--name", type=str, default="model")
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--num_steps", type=int, default=60)
parser.add_argument("--skip_step", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--checkpoint_period", type=int, default=2)
parser.add_argument("--save_best_only", action="store_true")
args = parser.parse_args()

data_path = os.path.abspath(args.data_path)
model_name = args.name
model_dir = os.path.abspath(args.model_dir)
if not os.path.isdir(model_dir):
    try:
        os.makedirs(model_dir)
    except:
        print("Error: the directory " + model_dir + " did not exist and could not be created")
        exit(1)
num_epochs = args.num_epochs
num_steps = args.num_steps
skip_step = args.skip_step
batch_size = args.batch_size
checkpoint_period = args.checkpoint_period
save_best_only = args.save_best_only

print("Using data from: " + data_path)
print("Saving models to: " + model_dir)

train_data, test_data, charset_size, reversed_dictionary = load_data.load_data(os.path.abspath(data_path))
train_data_generator = \
    load_data.KerasBatchGenerator(train_data, num_steps, batch_size, charset_size, skip_size=skip_step)
test_data_generator = \
    load_data.KerasBatchGenerator(test_data, num_steps, batch_size, charset_size, skip_size=skip_step)

hidden_size = 500
use_dropout = True
model = Sequential()
model.add(Embedding(charset_size, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(charset_size)))
model.add(Activation("softmax"))

optimizer = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])
print(model.summary())

checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir, model_name + "-{epoch:02d}.hdf5"),
                               verbose=1,
                               save_best_only=save_best_only,
                               period=checkpoint_period)
model.fit_generator(train_data_generator.generate(), len(train_data) // (batch_size * num_steps), num_epochs,
                    validation_data=test_data_generator.generate(),
                    validation_steps=len(test_data) // (batch_size * num_steps),
                    callbacks=[checkpointer])
model.save(os.path.join(model_dir, model_name + "-final.hdf5"))
with open(model_dir + r"\reversed_dictionary.pkl", "wb") as f:
    reversed_dictionary["message_delimiter"] = preprocess.message_delimiter
    pickle.dump(reversed_dictionary, f, pickle.HIGHEST_PROTOCOL)
