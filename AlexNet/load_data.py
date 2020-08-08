import os
import pickle

def unpickle(fp):
    load_dict = 0

    with open(fp, 'rb') as fid:
        load_dict = pickle.load(fid, encoding = 'bytes')

    return load_dict

