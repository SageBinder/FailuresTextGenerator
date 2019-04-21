import argparse
import os
import random

import numpy as np
import pandas as pd

message_delimiter = "<eom>"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("preprocessed_path", type=str)
    parser.add_argument("--user", type=str)
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv_path)
    user_to_train = args.user if args.user is not None else "everyone"
    username = user_to_train.replace("#", "")
    preprocessed_data_path = args.preprocessed_path

    print("Reading raw csv data from: " + csv_path)
    print("Extracting messages from: " + user_to_train)
    print("Saving preprocessed data to: " + preprocessed_data_path)

    chat = pd.read_csv(csv_path, usecols=["Author", "Content"], delimiter=";")
    if args.user is not None:
        chat = chat[chat["Author"] == user_to_train]
    chat = chat.dropna()

    # Append a message delimiter after each message
    for index, content in chat["Content"].items():
        chat.at[index, ["Content"]] = str(content) + message_delimiter

    total_data = chat["Content"].values
    random.seed(69)
    random.shuffle(total_data)

    train_data = total_data[0:int(0.8 * len(total_data))]
    test_data = total_data[int(0.8 * len(total_data)):]

    np.savetxt(os.path.join(preprocessed_data_path, "train.txt"), train_data, fmt="%s", encoding="utf-8")
    np.savetxt(os.path.join(preprocessed_data_path, "test.txt"), test_data, fmt="%s", encoding="utf-8")
    np.savetxt(os.path.join(preprocessed_data_path, "total.txt"), total_data, fmt="%s", encoding="utf-8")

    print(total_data)
