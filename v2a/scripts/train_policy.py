# v2a/scripts/train_policy.py
import os
import argparse
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from robomimic.config import config_factory, get_all_registered_configs
from v2a.algo.algo_policy import GMMPolicyAlgo
from v2a.configs import V2AConfig

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, obs_keys, action_key, frame_stack=1):
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.action_key = action_key
        self.demo_keys = []
        self.max_length = 150
        
        with h5py.File(hdf5_path, "r") as f:
            for demo in f["data"]:
                self.demo_keys.append(f"data/{demo}")
            print("In HDF5Dataset.__init__, self.demo_keys: ", self.demo_keys)

    def __len__(self):
        return len(self.demo_keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            demo = f[self.demo_keys[idx]]
            
            obs = {}
            # Load image observations
            obs["image"] = np.array(demo["obs/eye_in_hand_rgb"], dtype=np.float32) / 255.0
            obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))  # NHWC -> NCHW
            
            # Load robot states
            obs["robot_state"] = np.array(demo["robot_states"], dtype=np.float32)
            
            # Load actions
            actions = np.array(demo["actions"], dtype=np.float32)
            

            
                # 统一时间步长
        def pad_or_truncate(arr, target_length):
            """对输入的时间序列数据进行填充或截断，使其长度固定"""
            length = arr.shape[0]
            if length > target_length:
                return arr[:target_length]  # 截断
            elif length < target_length:
                pad_shape = (target_length - length, *arr.shape[1:])
                pad = np.zeros(pad_shape, dtype=arr.dtype)
                return np.concatenate([arr, pad], axis=0)  # 填充
            else:
                return arr  # 长度相同
            
                # 处理不同长度的时间步
        obs["image"] = pad_or_truncate(obs["image"], self.max_length)
        obs["robot_state"] = pad_or_truncate(obs["robot_state"], self.max_length)
        actions = pad_or_truncate(actions, self.max_length)
            
        assert obs["image"].shape[0] == obs["robot_state"].shape[0] == actions.shape[0], \
            f"Shape mismatch: {obs['image'].shape[0]}, {obs['robot_state'].shape[0]}, {actions.shape[0]}"
        # import pdb; pdb.set_trace()
            
        return {
            "obs": obs,
            "actions": actions
        }

def train(config, hdf5_path, output_dir):
    # Create dataset
    dataset = HDF5Dataset(
        hdf5_path=hdf5_path,
        obs_keys=["eye_in_hand_rgb", "robot_states"],
        action_key="actions"
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create algo
    print(get_all_registered_configs())
    algo_config = config_factory("bc")  # Using HBC as base config
    # import pdb; pdb.set_trace()
    algo = GMMPolicyAlgo(
        algo_config=algo_config.algo,
        obs_config=algo_config.observation,
        global_config=algo_config,
        obs_key_shapes={"eye_in_hand_rgb": [3, 128, 128], "robot_states": [9]},
        ac_dim=6,  # Match your action space
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Training loop
    for epoch in range(config.train.num_epochs):
        for batch in dataloader:
            # Process batch
            input_batch = algo.process_batch_for_training(batch)
            
            # Train step
            info = algo.train_on_batch(input_batch, epoch)
            
            # Logging
            print(f"Epoch {epoch}, Loss: {info['losses']['action_loss'].item()}")
            
        # Save checkpoint
        if epoch % config.train.save_interval == 0:
            torch.save(algo.serialize(), os.path.join(output_dir, f"model_{epoch}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to hdf5 dataset",
        default="/home/txs/Code/MimicPlay-Project/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to save checkpoints",
        default="/home/txs/Code/MimicPlay-Project/V2A/model_weights"
    )
    args = parser.parse_args()
    
    # # Create config
    # config = config_factory("hbc")  # Using HBC as base config
    # config.unlock()
    # config.train.batch_size = 32
    # config.train.num_epochs = 100
    # config.train.save_interval = 10
    
    CONFIG_PATH = "/home/txs/Code/MimicPlay-Project/V2A/v2a/configs/train_policy.json"
    ext_cfg = json.load(open(CONFIG_PATH, 'r'))
    config = V2AConfig(dict_to_load=ext_cfg)
    
    # Run training
    train(config, args.dataset, args.output_dir)