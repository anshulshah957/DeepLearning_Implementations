import os
import pickle
import numpy as np
from PIL import Image

def unpickle(fp):
    load_dict = 0

    with open(fp, 'rb') as fid:
        load_dict = pickle.load(fid, encoding = 'bytes')

    return load_dict


'''
https://github.com/amir-saniyan/AlexNet/blob/master/dataset_helper.py
'''

def get_cifar_10(image_width, image_height, cifar_path = '../cifar-10-batches-py'):
    batch_1 = unpickle(os.path.abspath(os.path.join(cifar_path, 'data_batch_1')))
    batch_2 = unpickle(os.path.abspath(os.path.join(cifar_path, 'data_batch_2')))
    batch_3 = unpickle(os.path.abspath(os.path.join(cifar_path, 'data_batch_3')))
    batch_4 = unpickle(os.path.abspath(os.path.join(cifar_path, 'data_batch_4')))
    batch_5 = unpickle(os.path.abspath(os.path.join(cifar_path, 'data_batch_5')))

    test_batch = unpickle(os.path.abspath(os.path.join(cifar_path, 'test_batch')))

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_samples = len(batch_1[b'labels']) + len(batch_2[b'labels']) + len(batch_3[b'labels'])\
                          + len(batch_4[b'labels']) + len(batch_5[b'labels'])

    X_train = np.zeros(shape = [train_samples, image_width, image_height, 3], dtype=np.uint8)
    Y_train = np.zeros(shape=[train_samples, len(classes)], dtype=np.float32)

    batches = [batch_1, batch_2, batch_3, batch_4, batch_5]

    index = 0
    for batch in batches:
        for i in range(len(batch[b'labels'])):
            image = batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
            label = batch[b'labels'][i]

            X = np.array(Image.fromarray(image).resize((image_width, image_height)))
            Y = np.zeros(shape=[len(classes)], dtype=np.int)
            Y[label] = 1

            X_train[index + i] = X
            Y_train[index + i] = Y

        index += len(batch[b'labels'])

    test_samples = len(test_batch[b'labels'])

    X_test = np.zeros(shape = [test_samples, image_width, image_height, 3], dtype=np.uint8)
    Y_test = np.zeros(shape = [test_samples, len(classes)], dtype=np.float32)

    for i in range(len(test_batch[b'labels'])):
        image = test_batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
        label = test_batch[b'labels'][i]

        X = np.array(Image.fromarray(image).resize((image_width, image_height)))
        Y = np.zeros(shape=[len(classes)], dtype=np.int)
        Y[label] = 1

        X_test[i] = X
        Y_test[i] = Y

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    x, _, _, _ = get_cifar_10(70, 70)
    print(x.shape)

