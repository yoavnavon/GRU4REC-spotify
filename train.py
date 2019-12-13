from keras.utils import to_categorical
import keras.backend as K
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import SessionDataset, SessionDataLoader
from model import GRU4REC


def get_states(model):
    return [K.get_value(s) for s, _ in model.state_updates]


def get_metrics(model, loader, args, k=20):

    rec_sum = 0.0
    mrr_sum = 0.0
    rec_sum_alt = 0.0
    mrr_sum_alt = 0.0
    batch = 0
    batch_size = args.batch_size
    n_items = loader.dataset.n_items
    tracks_feats = pd.read_csv('tracks_feats.csv', index_col=0)
    for feat, label, mask in loader:
        # [batch_size, n_classes]
        target_oh = to_categorical(label, num_classes=n_items)
        if args.input_form == 'one-hot':
            # [batch_size, n_classes]
            input_oh = to_categorical(feat,  num_classes=n_items)
            # [batch_size, 1, n_clasess]
            input_oh = np.expand_dims(input_oh, axis=1)
            feat = input_oh
        if args.input_form.startswith('content'):
            feat = tracks_feats.loc[feat, :].values
            feat = np.expand_dims(feat, axis=1)

        # [batch_size, n_classes]
        pred = model.predict(feat, batch_size=batch_size)

        # get values, index pairs
        partition = np.partition(pred, -k, axis=1)[:, -k:]
        arg_partition = np.argpartition(pred, -k, axis=1)[:, -k:]
        stack = np.stack([partition, arg_partition], axis=2)

        # sorted values indexes
        index = stack[:, :, 0].argsort(axis=1)[:, :, None]

        # take based on sorted values indexes
        topk = np.take_along_axis(stack, index, axis=1)[:, :, 1].astype(int)

        # relevant mask
        mask = np.repeat(label, k).reshape(batch_size, k) == topk

        # weights for mrr
        weights = np.arange(1, k + 1)

        # compute based on relevant and weights (mrr only)
        rec_sum_alt += mask.sum()
        mrr_sum_alt += (mask / weights).sum()
        batch += 1

        if batch == args.n_batch_validate:
            break

    recall = rec_sum_alt / (batch * batch_size)
    mrr = mrr_sum_alt / (batch * batch_size)
    return (recall, mrr)


def train_model(args):
    dataset = SessionDataset(
        args.data_path,
        args.idxs_path,
        session_key=args.session_key,
        item_key=args.item_key)

    test_dataset = SessionDataset(
        args.test_path,
        args.idxs_path,
        session_key=args.session_key,
        item_key=args.item_key)

    test_loader = SessionDataLoader(test_dataset, batch_size=args.batch_size)

    model_to_train = GRU4REC(args, dataset.n_items)
    batch_size = args.batch_size

    epoch = 0
    batch = 0
    rnn_idx = 1
    if args.input_form == 'emb':
        rnn_idx = 2
    if args.input_form == 'content-mlp':
        rnn_idx = 3

    tracks_feats = pd.read_csv('tracks_feats.csv', index_col=0)

    for epoch in range(1, args.epochs):
        t = tqdm(total=args.validate_batch)
        loader = SessionDataLoader(dataset, batch_size=batch_size)
        for feat, target, mask in loader:
            real_mask = np.ones((batch_size, 1))
            for elt in mask:
                real_mask[elt, :] = 0
            hidden_states = get_states(model_to_train.model)[0]
            hidden_states = np.multiply(real_mask, hidden_states)
            hidden_states = np.array(hidden_states, dtype=np.float32)
            model_to_train.model.layers[rnn_idx].reset_states(hidden_states)
            if args.input_form == 'one-hot':
                input_oh = to_categorical(feat, num_classes=dataset.n_items)
                input_oh = np.expand_dims(input_oh, axis=1)
                feat = input_oh
            if args.input_form.startswith('content'):
                feat = tracks_feats.loc[feat, :].values
                feat = np.expand_dims(feat, axis=1)
            if args.loss == 'crossentropy':
                target = to_categorical(target, num_classes=dataset.n_items)

            tr_loss = model_to_train.model.train_on_batch(feat, target)

            batch += 1
            t.set_description(
                "Epoch {0}. Batch {1}. Loss: {2:.5f}".format(epoch, batch, tr_loss))
            t.update(1)

            if not batch % args.validate_batch:
                print("\nEvaluating Model")
                rec, mrr = get_metrics(model_to_train.model, test_loader, args)
                print("Recall@{}: {:5f}".format(args.topk, rec))
                print("MRR@{}: {:5f}".format(args.topk, mrr))
                print("\n")
                t.close()
                t = tqdm(total=args.validate_batch)
                # t.reset()


def evaluate_most_popular(args):
    model = MostPopular()
    test_dataset = SessionDataset(
        args.test_path,
        args.idxs_path,
        session_key=args.session_key,
        item_key=args.item_key)

    test_loader = SessionDataLoader(test_dataset, batch_size=args.batch_size)

    rec, mrr = get_metrics(model, test_loader, args)
    print("Recall@{}: {:5f}".format(args.topk, rec))
    print("MRR@{}: {:5f}".format(args.topk, mrr))
    print("\n")
