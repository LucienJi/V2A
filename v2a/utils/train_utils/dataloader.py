
import os
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

from dataset import PlaydataSequenceDataset

def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (PlaydataSequenceDataset instance): train dataset object
        valid_dataset (PlaydataSequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        train_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=train_filter_by_attribute,
        )
        valid_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=valid_filter_by_attribute,
        )
        assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
            "validation demonstrations!"
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a PlaydataSequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the PlaydataSequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (PlaydataSequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=config.train.dataset_keys,
        goal_obs_gap=config.algo.playdata.goal_image_range,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute
    )
    dataset = PlaydataSequenceDataset(**ds_kwargs)

    return dataset