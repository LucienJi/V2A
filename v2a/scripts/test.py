
import sys,os,json
import numpy as np 
import torch 
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger
from v2a.configs import V2AConfig
from v2a.utils.train_utils.dataloader import load_data_for_encoder_training
from torch.utils.data import DataLoader
from v2a.algo.algo_encoder import EncoderAlgo

CONFIG_PATH = "/code/V2A/v2a/configs/train_encoder.json"
def train():
    ext_cfg = json.load(open(CONFIG_PATH, 'r'))
    config = V2AConfig(dict_to_load=ext_cfg)
    # ObsUtils.initialize_obs_utils_with_config(config)
    ObsUtils.initialize_obs_utils_with_obs_specs([config.observation.modalities])

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    print(shape_meta['all_shapes'])
    all_shapes = shape_meta['all_shapes']
    algo = EncoderAlgo(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=shape_meta['all_shapes'],
        ac_dim=0,
        device='cuda'
    )

    # trainset = load_data_for_encoder_training(
    #     config, obs_keys=shape_meta["all_obs_keys"])
    # train_sampler = trainset.get_dataset_sampler()
    # train_loader = DataLoader(
    #     dataset=trainset,
    #     sampler=train_sampler,
    #     batch_size=config.train.batch_size,
    #     shuffle=(train_sampler is None),
    #     num_workers=config.train.num_data_workers,
    #     drop_last=True
    # )

    # data_loader_iter = iter(train_loader)
    


    

if __name__ == "__main__":
    train()