import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler


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


def get_track_feats(path):
    tracks0 = pd.read_csv(path + 'track_features/tf_000000000000.csv')
    tracks1 = pd.read_csv(path + 'track_features/tf_000000000001.csv')
    items_idxs = json.load(open('item_idxs.json', 'r'))

    mask = []
    for i in tracks0['track_id']:
        mask.append(i in items_idxs)
    mask0 = np.array(mask)
    tracks0 = tracks0[mask0]

    mask = []
    for i in tracks1['track_id']:
        mask.append(i in items_idxs)
    mask1 = np.array(mask)
    tracks1 = tracks1[mask1]

    track_feats = pd.concat([tracks0, tracks1])
    track_feats = track_feats.drop('mode', axis=1)
    track_feats['idx'] = track_feats['track_id'].apply(lambda x: items_idxs[x])
    track_feats = track_feats.drop('track_id', axis=1)
    track_feats = track_feats.set_index('idx')

    track_feats_std = pd.DataFrame(StandardScaler().fit_transform(
        track_feats), index=track_feats.index)
    track_feats_std.to_csv('tracks_feats.csv', index=True)


if __name__ == "__main__":
    training_file = 'data/training/log_3_20180827_000000000000.csv.gz'
    testing_file = 'data/log_1_20180715_000000000000.csv.gz'
    print('creating test file...')
    create_test_split(training_file, testing_file)

    input_path = 'data/training'
    print('creating item_idxs...')
    get_item_idxs(input_path)

    print('creating track feats data...')
    get_track_feats('data/')
