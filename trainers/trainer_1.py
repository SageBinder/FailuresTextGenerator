import os
import pickle
import sys

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from data_loader import load_data
from datasets import preprocess

data_path = os.path.abspath(os.path.join(r"..\datasets", sys.argv[1]))  # argv[1] example: "saggisponge1234\trash"

# example models_path will be: "..\models\trainer_1\saggisponge1234\trash
models_path = os.path.abspath(os.path.join(r"..\models\trainer_1", sys.argv[1]))
if not os.path.isdir(models_path):
    try:
        os.makedirs(models_path)
    except:
        print("Error: could not make model directory " + models_path)
        exit(1)

print("Using data from: " + data_path)
print("Saving models to: " + models_path)

num_steps = 60
skip_step = num_steps
batch_size = 256
train_data, test_data, charset_size, reversed_dictionary = \
    load_data.load_data(os.path.abspath(data_path))

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

num_epochs = 1
checkpointer = ModelCheckpoint(filepath=models_path + r"\model-{epoch:02d}.hdf5", verbose=1)
model.fit_generator(train_data_generator.generate(), len(train_data) // (batch_size * num_steps), num_epochs,
                    validation_data=test_data_generator.generate(),
                    validation_steps=len(test_data) // (batch_size * num_steps),
                    callbacks=[checkpointer])
model.save(models_path + r"\final_model.hdf5")
with open(models_path + r"\reversed_dictionary.pkl", "wb") as f:
    reversed_dictionary["message_delimiter"] = preprocess.message_delimiter
    pickle.dump(reversed_dictionary, f, pickle.HIGHEST_PROTOCOL)
