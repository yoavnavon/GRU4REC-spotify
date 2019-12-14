from easydict import EasyDict
from train import train_model

args = EasyDict({
    'batch_size': 64,
    'epochs': 100,
    'validate_epoch': 1,
    'validate_batch': 3000,
    'n_batch_validate': 500,
    'emb_size': 64,
    'hidden_units': 100,
    'lr': 0.001,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'dropout': 0.1,
    'topk': 20,
    'loss': 'bpr-max',  # [crossentropy, bpr, bpr-max]
    'activation': 'linear',
    'session_key': 'session_id',
    'item_key': 'track_id_clean',
    'data_path': 'data/training',
    'test_path': 'data/testing',
    'idxs_path': 'item_idxs.json',
    'input_form': 'content',  # [one-hot, emb, content, content-mlp]
    'regularization': None,
})

train_model(args)
