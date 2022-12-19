from __future__ import print_function

import os
import numpy as np
import sys

from skimage.io import imsave, imread

data_path = 'raw/'

image_rows  = 256
image_cols  = 256

if len(sys.argv) > 1:
    input_type = sys.argv[1]
else:
    input_type = "infarct"

if len(sys.argv) > 2:
    input_type = sys.argv[1] + '_all'


def create_train_data():
    print("create_train_data: ", input_type)
    train_data_path = os.path.join(data_path, '{:s}_train'.format(input_type))
    images = os.listdir(train_data_path + '/img')
    total = len(images)
    print("total ",input_type,": ", total)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        image_mask_name = 'mask_' + image_name.split('_')[1]
        img = imread(os.path.join(train_data_path + '/img', image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path + '/mask', image_mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('{:s}_imgs_train.npy'.format(input_type), imgs)
    np.save('{:s}_imgs_mask_train.npy'.format(input_type), imgs_mask)
    print('Saving to .npy files done.')


def load_train_data(data_type):
    imgs_train = np.load('{:s}_imgs_train.npy'.format(data_type))
    imgs_mask_train = np.load('{:s}_imgs_mask_train.npy'.format(data_type))
    return imgs_train, imgs_mask_train


def create_test_data():
    print("create_test_data: ", input_type)
    train_data_path = os.path.join(data_path, '{:s}_test'.format(input_type))
    print("train_data_path: ", train_data_path)
    images = os.listdir(train_data_path + '/img')
    total = len(images)
    print("total ",input_type,": ", total)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)

    for image_name in images:
        img_id = int((image_name.split('_')[1]).split('.')[0])
        img = imread(os.path.join(train_data_path + '/img', image_name), as_gray=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('{:s}_imgs_test.npy'.format(input_type), imgs)
    np.save('{:s}_imgs_id_test.npy'.format(input_type), imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_type):
    imgs_test = np.load('{:s}_imgs_test.npy'.format(data_type))
    imgs_id = np.load('{:s}_imgs_id_test.npy'.format(data_type))
    return imgs_test, imgs_id

if __name__ == '__main__':

    if len(sys.argv) <= 2:
        create_train_data()
    create_test_data()
    