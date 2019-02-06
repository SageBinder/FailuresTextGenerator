import os
import random
import sys

import numpy as np
import pandas as pd

message_delimiter = "<eom>"

if __name__ == "__main__":
    raw_csv_data_path = os.path.abspath(sys.argv[1])       # argv[1] example: "raw_csv\trash.csv
    user_to_train = sys.argv[2]                            # argv[2] example: "saggisponge#1234"
    username = user_to_train.replace("#", "")

    # example preprocessed_data_path will be: "saggisponge1234\trash"
    preprocessed_data_path = \
        os.path.join(os.path.abspath(username), os.path.split(os.path.splitext(raw_csv_data_path)[0])[1])

    print("Reading raw csv data from: " + raw_csv_data_path)
    print("Extracting messages from: " + user_to_train)
    print("Saving preprocessed data to: " + preprocessed_data_path)

    chat = pd.read_csv(raw_csv_data_path, usecols=["Author", "Content"], delimiter=";")
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
