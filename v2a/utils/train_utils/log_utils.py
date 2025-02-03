import os
import time
import datetime
import shutil
import v2a

def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(v2a.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)
    return log_dir, output_dir, video_dir