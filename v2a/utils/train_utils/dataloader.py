
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

from .dataset import EncoderSequenceDataset


def load_data_for_encoder_training(config, obs_keys):
    """
    Data loading at training time for the encoder.
    No need to set training and validation split here, since encoder is trained on the entire dataset.
    """

    # config can contain an attribute to filter on
    dataset = encoder_dataset_factory(config, obs_keys)
    return dataset

def encoder_dataset_factory(config, obs_keys, dataset_path=None):
    
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
    )
    dataset = EncoderSequenceDataset(**ds_kwargs)

    return dataset