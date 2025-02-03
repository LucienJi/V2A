import hydra
from omegaconf import DictConfig, OmegaConf
import sys,os
import numpy as np 
import torch 
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger


from v2a.utils import get_exp_dir
@hydra.main(config_path="../configs", config_name="debug.yaml")
def train(config: DictConfig):
    # Print the loaded configuration for debugging purposes.
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    print("\n============= New Training Run with Config =============")

    log_dir, ckpt_dir, video_dir = get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    

if __name__ == "__main__":
    train()