import argparse
import os
import pickle

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from data_loader import load_data
from datasets import preprocess

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("model_path", type=str)
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
model_path = args.model_path
model_dir = os.path.dirname(model_path)
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

train_data, test_data, charset_size, reversed_dictionary = load_data.load_data(os.path.abspath(data_path))
train_data_generator = \
    load_data.KerasBatchGenerator(train_data, num_steps, batch_size, charset_size, skip_size=skip_step)
test_data_generator = \
    load_data.KerasBatchGenerator(test_data, num_steps, batch_size, charset_size, skip_size=skip_step)

model = load_model(model_path)
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
