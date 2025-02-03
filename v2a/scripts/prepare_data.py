import hydra
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from omegaconf import DictConfig
from utils.env_utils.robomimic_states_to_obs import robomimic_states_to_obs
import libero.libero.envs 

@hydra.main(config_path="../configs", config_name="robosuite_dataset.yaml",version_base=None)
def convert_robomimic_state(config: DictConfig):
    robomimic_states_to_obs(config)

if __name__ == "__main__":
    convert_robomimic_state()
    