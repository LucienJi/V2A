
import sys,os,json
import numpy as np 
import torch 
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger
from v2a.configs import V2AConfig
from v2a.utils.train_utils.dataloader import encoder_dataset_factory
from torch.utils.data import DataLoader
from v2a.algo.algo_encoder import EncoderAlgo

CONFIG_PATH = "/code/V2A/v2a/configs/debug.json"
def train():
    ext_cfg = json.load(open(CONFIG_PATH, 'r'))
    config = V2AConfig(dict_to_load=ext_cfg)
    ObsUtils.initialize_obs_utils_with_obs_specs([config.observation.modalities])

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    algo = EncoderAlgo(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=shape_meta['all_shapes'],
        ac_dim=0,
        device='cuda'
    )

    trainset = encoder_dataset_factory(
        config, obs_keys=shape_meta["all_obs_keys"])
    
    algo.set_train()

    total_epochs = 5
    n_traj_per_gradient = 2
    n_steps_per_epoch = trainset.n_demos// n_traj_per_gradient
    for epoch in range(total_epochs):
        demos_index_permuted = np.random.permutation(trainset.n_demos)
        idx = 0
        for _ in range(n_steps_per_epoch):
            demo_index = demos_index_permuted[idx%len(demos_index_permuted)]
            batch = trainset.get_batch_within_trajectory(demo_index,bz = config.train.batch_size)
            
            import pdb; pdb.set_trace()
            batch = TensorUtils.to_torch(batch, device='cuda')
            batch = algo.process_batch_for_training(batch)
            info = algo.train_on_batch(batch,epoch)
            print(info)
            idx += 1


def train_policy():
    from v2a.algo.algo_policy import GMMPolicyAlgo
    from v2a.utils.train_utils.dataloader import policy_dataset_factory
    ext_cfg = json.load(open(CONFIG_PATH, 'r'))
    config = V2AConfig(dict_to_load=ext_cfg)
    ObsUtils.initialize_obs_utils_with_obs_specs([config.observation.modalities])

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    algo = GMMPolicyAlgo(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=shape_meta['all_shapes'],
        ac_dim=config.algo.policy.action_dim,
        device=config.train.device
    )

    trainset = policy_dataset_factory(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    dataloader = DataLoader(trainset, batch_size=3, shuffle=True)
    algo.set_train()

    # Training loop
    for epoch in range(config.train.num_epochs):
        for batch in dataloader:
            # Process batch
            batch = TensorUtils.to_torch(batch, device=config.train.device)
            input_batch = algo.process_batch_for_training(batch)
            
            # Train step
            info = algo.train_on_batch(input_batch, epoch)
            
            # Logging
            print(f"Epoch {epoch}, Loss: {info['losses']['action_loss'].item()}")
            
        # Save checkpoint
        if epoch % config.train.save_interval == 0:
            pass
            # torch.save(algo.serialize(), os.path.join(output_dir, f"model_{epoch}.pt")) 


    

    
    


    

if __name__ == "__main__":
    train_policy()