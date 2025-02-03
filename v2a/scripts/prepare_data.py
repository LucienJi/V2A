import hydra
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from omegaconf import DictConfig
from utils.env_utils.dataset_states_to_obs3d import dataset_states_to_obs
import libero.libero.envs 

@hydra.main(config_path="../configs", config_name="robosuite_dataset.yaml",version_base=None)
def main(config: DictConfig):
    dataset_states_to_obs(config)

if __name__ == "__main__":
    main()
    