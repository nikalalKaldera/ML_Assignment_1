import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sklearn.model_selection import train_test_split


def analyse_data_structure(dataframe):
    # analyse data structure
    # print(dataframe.head())  # print some data
    # print(dataframe.info())  # Description of dataset
    # print(dataframe["pH"].value_counts())  # counts of each quality
    # pd.set_option('display.max_columns', None)  # display all the columns   #summary of attributes
    print(dataframe.describe())
    dataframe.hist(bins=50, figsize=(20, 15))  # plot histogram for attributes
    plt.savefig(fname="his.png")
    plt.show()
    return


def choose_fixed_indices(df):
    # Use the hash of the DataFrame's contents as the seed value
    hash_value = hashlib.md5(df.to_numpy().tobytes()).hexdigest()
    seed_value = int(hash_value, 16) % 2 ** 32

    # Use the fixed seed value to select a subset of row indices from the DataFrame
    rng = np.random.default_rng(seed_value)
    indices = rng.choice(df.index, size=5, replace=False)

    # Create a new DataFrame with the selected rows removed
    selected_rows = df.loc[indices]
    new_df = df.drop(indices)

    return selected_rows, new_df

if __name__ == '__main__':
    # load wine dataset
    data = pd.read_csv("winequality-red.csv", delimiter=';')
    # analyse_data_structure(data)
    model_test_data, train_data = choose_fixed_indices(data)
    # print(train_data.head())
    # pd.set_option('display.max_columns', None)
    # print(model_test_data.head())
    print("Length of train data :", len(train_data))
    print("Length of model testing data", len(model_test_data))
    train_set, test_set = train_test_split(train_data, test_size=0.2, random_state=42)
    print(train_set.head())
    print("Length of final model train data", len(train_set))
    print(test_set.head())
    print("Length of final test_data", len(test_set))

