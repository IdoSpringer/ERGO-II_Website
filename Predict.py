import numpy as np
import pickle
from Loader import SignedPairsDataset, get_index_dicts
from Trainer import ERGOLightning
from Sampler import read_input_file
from torch.utils.data import DataLoader
from argparse import Namespace
import torch
import pandas as pd
import os
from os import listdir
from os.path import isfile, join


def load_model(hparams, checkpoint_path):
    model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_model_from_version(version):
    # get model file from version
    model_dir = 'paper_models'
    logs_dir = 'ERGO-II_paper_logs'
    checkpoint_path1 = os.path.join(model_dir, 'version_' + version, 'checkpoints')
    checkpoint_path2 = os.path.join(logs_dir, checkpoint_path1)
    try:
        files = [f for f in listdir(checkpoint_path1) if isfile(join(checkpoint_path1, f))]
        checkpoint_path = checkpoint_path1
    except FileNotFoundError:
        files = [f for f in listdir(checkpoint_path2) if isfile(join(checkpoint_path2, f))]
        checkpoint_path = checkpoint_path2
    checkpoint_path = os.path.join(checkpoint_path, files[0])
    # get args from version
    args_path1 = os.path.join(logs_dir, model_dir, 'version_' + version)
    args_path2 = os.path.join(logs_dir, model_dir, version)
    if isfile(os.path.join(args_path1, 'meta_tags.csv')):
        args_path = os.path.join(args_path1, 'meta_tags.csv')
    elif isfile(os.path.join(args_path2, 'meta_tags.csv')):
        args_path = os.path.join(args_path2, 'meta_tags.csv')
    with open(args_path, 'r') as file:
        lines = file.readlines()
        args = {}
        for line in lines[1:]:
            key, value = line.strip().split(',')
            if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
                args[key] = value
            else:
                args[key] = eval(value)
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
    test_pickle = 'Samples/' + model.dataset + '_test_samples.pickle'
    datafiles = train_pickle, test_pickle
    return model, train_pickle


def get_train_dicts(train_pickle):
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    return train_dicts


def predict(version, test_file):
    model, train_file = get_model_from_version(version)
    train_dicts = get_train_dicts(train_file)
    test_samples, dataframe = read_input_file(test_file)
    test_dataset = SignedPairsDataset(test_samples, train_dicts)
    batch_size = 1000
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=model.tcr_encoding_model,
                                                                  cat_encoding=model.cat_encoding))
    outputs = []
    for batch_idx, batch in enumerate(loader):
        output = model.validation_step(batch, batch_idx)
        if output:
            outputs.extend(output['y_hat'].tolist())
    dataframe['Score'] = outputs
    return dataframe


if __name__ == '__main__':
    # df = predict('1meaj', 'example2.csv')
    # print(df)
    # df.to_csv('results.csv', index=False)
    pass


# NOTE: fix sklearn import problem with this in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/
# or just conda install libgcc
