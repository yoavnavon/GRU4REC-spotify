from glob import glob
import numpy as np
import json
import pandas as pd


class SessionDataset:

    def __init__(self, data_path, idxs_path, session_key=None, item_key=None):
        self.filenames = glob(f'{data_path}/*.csv.gz')
        self.item_idxs = json.load(open(idxs_path))
        self.session_key = session_key
        self.item_key = item_key
        self.n_items = len(self.item_idxs)
        self.data_path = data_path


class SessionDataLoader:
    def __init__(self, dataset, batch_size):
        self.filenames = dataset.filenames[:]
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader = None

    def __iter__(self):
        dataset = self.dataset
        filenames = self.filenames
        while filenames or self.loader:
            if not self.loader:
                filename = filenames.pop()
                df = pd.read_csv(filename)
                df = df[['session_id', 'track_id_clean']]
                dataset = SessionDatasetMini(
                    df,
                    session_key=dataset.session_key,
                    item_key=dataset.item_key,
                    item_idxs=dataset.item_idxs)
                loader = SessionDataLoaderMini(
                    dataset, batch_size=self.batch_size)
                self.loader = loader
            loader = self.loader
            for feat, target, mask in loader:
                yield feat, target, mask
            self.loader = None
            del loader
            del dataset
            del df


class SessionDatasetMini:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, data, session_key='SessionId', item_key='ItemId', item_idxs=None):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        self.df = data
        self.session_key = session_key
        self.item_key = item_key
        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = self.order_session_idx()
        self.item_idxs = item_idxs
        self.add_item_idxs()

    def add_item_idxs(self):
        self.df['item_idx'] = self.df[self.item_key].apply(
            lambda x: self.item_idxs[x])

    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(
            self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

        return offsets

    def order_session_idx(self):
        """ Order the session indices """
        session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr


class SessionDataLoaderMini:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.
        Args:
            dataset (SessionDataset): the session dataset to generate the batches from
            batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0
        self.sum = 0

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """
        df = self.dataset.df
        session_key = self.dataset.session_key
        item_key = self.dataset.item_key
        self.n_items = df[item_key].nunique()+1
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        # start idx of every session
        start = click_offsets[session_idx_arr[iters]]
        # end idx of every session
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()  # Minimum session length
            # Item indices (for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]

            for i in range(minlen):
                if i >= 1:
                    self.done_sessions_counter = 0
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                inp = idx_input
                target = idx_target
                yield inp, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)

            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            self.done_sessions_counter = len(mask)
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
