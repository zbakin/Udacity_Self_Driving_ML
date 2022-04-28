import argparse
import glob
import os
import random

import numpy as np
import shutil
from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # get dataset
    dataset = glob.glob(source+"/*.tfrecord")
    # get number of files
    data_size = len(dataset)
    # shuffle the dataset
    np.random.shuffle(dataset)
    # split training 80% and validation 20%
    train, val = np.split(dataset, [int(0.8 * data_size)])

    # prepare train and val directory paths
    train_dir = os.path.join(destination, "train/")
    val_dir = os.path.join(destination, "val/")

    # move train data to train folder
    for file in train:
        shutil.move(file, train_dir)

    # move val data to val folder
    for file in val:
        shutil.move(file, val_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)