import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import json


def create_test_split(training_file, testing_file):
    train = pd.read_csv(training_file)
    test = pd.read_csv(testing_file)
    mask = item_is_in(test, train)
    test = test[mask]
    basename = os.path.basename(testing_file)
    test.to_csv(f'data/testing/{basename}')
    del train
    del test
    del mask


def item_is_in(test, train):
    items = set()
    for i in train.track_id_clean.unique():
        items.add(i)
    mask = []
    for i in test.track_id_clean:
        mask.append(i in items)
    del items
    return np.array(mask)


def get_item_idxs(input_path):
    items = {}
    item_idx = 0
    for path in tqdm(glob(f'{input_path}/*.csv.gz')):
        df = pd.read_csv(path)
        for track in df.track_id_clean.unique():
            if track not in items:
                items[track] = item_idx
                item_idx += 1
        del df
    with open('item_idxs.json', 'w') as file:
        json.dump(items, file)


if __name__ == "__main__":
    training_file = 'data/training/log_3_20180827_000000000000.csv.gz'
    testing_file = 'data/log_1_20180715_000000000000.csv.gz'
    create_test_split(training_file, testing_file)

    input_path = 'data/training'
    get_item_idxs(input_path)
