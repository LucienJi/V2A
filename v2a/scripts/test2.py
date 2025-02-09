import numpy as np
from v2a.utils.train_utils.dataloader import load_data_for_encoder_training
import json 
from v2a.configs import V2AConfig
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils


CONFIG_PATH = "/code/V2A/v2a/configs/train_encoder.json"
ext_cfg = json.load(open(CONFIG_PATH, 'r'))
config = V2AConfig(dict_to_load=ext_cfg)
ObsUtils.initialize_obs_utils_with_obs_specs([config.observation.modalities])

shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
trainset = load_data_for_encoder_training(
        config, obs_keys=shape_meta["all_obs_keys"])
## this is a test code for a new branch.

if __name__ == "__main__":
    batch = trainset.get_batch_within_trajectory(0,16)
    for k,v in batch.items():
        if k == 'obs':
            for k2,v2 in v.items():
                print(k2,v2.shape)
        else:
            print(k,v.shape)