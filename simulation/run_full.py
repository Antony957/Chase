import os
import sys
import h5py
import json
import logging
import argparse
import subprocess
import numpy as np
from easydict import EasyDict

DP_SRC_ROOT = os.path.join(os.path.dirname(__file__), "../src")
print("diffusion policy source root:", DP_SRC_ROOT)
sys.path.insert(0, DP_SRC_ROOT)

from extract_demo import extract_demo
from robomimic_dataset_conversion import convert_rel_actions_to_abs
from extract_useful_data import extract_useful_data
from dataset_combine import combine_dataset

def generate_training_cfg(task, used_demo, dataset_path, abs_output_dir, seeds, prefix):
    config_template_path = os.path.join("config_template/training", task + ".json")
    assert os.path.exists(config_template_path)
    with open(config_template_path, "r") as f:
        config_template = json.load(f)
    config_template = EasyDict(config_template)
    for seed in seeds:
        config = config_template.copy()
        config = EasyDict(config)
        config.seed = seed
        config.dataset.params.path = dataset_path
        config.dataset.params.train_filter_key = used_demo
        config.log_dir = os.path.join(abs_output_dir, "logs/{}_seed_{}".format(prefix, seed))
        config.resume_ckpt = None
        config.resume_epoch = -1
        config_path = os.path.join(abs_output_dir, "configs/{}_seed_{}.json".format(prefix, seed))
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logging.info("Generated training config {}_seed_{}.json".format(prefix, seed))
        with h5py.File(dataset_path, "r") as f:
            num_demos = len(f["mask/{}".format(used_demo)])
            logging.info("Number of demos in {}: {}".format(used_demo, num_demos))

def start_training(abs_output_dir, seeds, prefix, adv=False, cuda_device=None):
    processes = []
    for i, seed in enumerate(seeds):
        config_name = "{}_seed_{}.json".format(prefix, seed)
        visible_devices = 0 if cuda_device is None else cuda_device[i%len(cuda_device)]
        # cmd_prefix = "torchrun --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node=gpu train.py "
        cmd_prefix = "python train_single_gpu.py "
        cmd = "cd {} && CUDA_VISIBLE_DEVICES={} ".format(DP_SRC_ROOT, visible_devices) + \
              cmd_prefix + \
              "--config {}".format(os.path.join(abs_output_dir, "configs/{}".format(config_name)))
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        logging.info("Started training {}".format(config_name))
    # spin until all training is done
    for process in processes:
        process.wait()
    logging.info("All training is done")

def start_evaluating(abs_output_dir, seeds, prefix, eval_specific_args="", eval_output_dir_name="eval", count_only=False, cuda_device=None, adv=False):
    processes = []
    eval_log_dir_dict = {}
    for i, seed in enumerate(seeds):
        config_name = "{}_seed_{}.json".format(prefix, seed)
        trys = os.listdir(os.path.join(abs_output_dir, "logs/{}_seed_{}".format(prefix, seed)))
        trys = sorted(trys) # use the last try
        eval_dataset_id = "{}_seed_{}".format(prefix, seed)
        eval_log_id = trys[-1]
        eval_log_dir = os.path.join(abs_output_dir, "logs", eval_dataset_id, eval_log_id)
        eval_log_dir_dict[eval_dataset_id] = eval_log_dir
        if not count_only:
            eval_ckpt = "policy_last"
            rollout_args = "--n_rollouts 200 --seed 233 --try_times 5 --abs_action"
            dataset_name = "demo.hdf5"
            if adv:
                rollout_args += " --enable_adv"
                dataset_name = "demo_adv.hdf5"

            extra_args = "--dataset_obs --render_traj --return_intermediate"
            cmd = "CUDA_VISIBLE_DEVICES={} python run.py --agent {} --config {} {} --dataset_path {}/{} --video_dir {} {} {}".format(
                0 if cuda_device is None else cuda_device[i%len(cuda_device)],
                os.path.join(abs_output_dir, "logs", eval_dataset_id, eval_log_id, "ckpt", eval_ckpt + ".ckpt"),
                os.path.join(abs_output_dir, "logs", eval_dataset_id, eval_log_id, "config.json"),
                rollout_args,
                os.path.join(eval_log_dir, eval_output_dir_name),
                dataset_name,
                os.path.join(eval_log_dir, eval_output_dir_name),
                extra_args,
                eval_specific_args
            )
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)
            logging.info("Started evaluating {} {}".format(eval_dataset_id, eval_specific_args))
    # spin until all evaluation is done
    for process in processes:
        process.wait()
    logging.info("All evaluation is done")
    # calculate success rate
    success_rate_list = []
    for seed in seeds:
        eval_demo_path = os.path.join(eval_log_dir_dict["{}_seed_{}".format(prefix, seed)], eval_output_dir_name, "demo.hdf5")
        success_num = 0
        with h5py.File(eval_demo_path, "r") as f:
            for ep in f["data"]:
                is_success = np.any(f["data/{}/dones".format(ep)][()])
                if is_success:
                    success_num += 1
            success_rate = success_num / len(f["data"])
            success_rate_list.append(success_rate)
        logging.info("Success rate of {}_seed_{}/{}/{}: {}".format(prefix, seed, eval_log_id, eval_output_dir_name, success_rate))
    logging.info("Success rate mean: {}".format(np.mean(success_rate_list)))
    logging.info("Success rate std: {}".format(np.std(success_rate_list)))
    return eval_log_dir_dict

def start_extracting_useful_data(eval_path, seeds, prefix, eval_output_dir_name="eval"):
    for seed in seeds:
        eval_demo_path = os.path.join(eval_path, eval_output_dir_name, "demo_adv.hdf5")
        print(eval_demo_path)
        assert os.path.exists(eval_demo_path)
        same_init_state_repeated_times = 5
        extract_useful_data(eval_demo_path, "success", same_init_state_repeated_times, threshold=1.1)
        extract_useful_data(eval_demo_path, "sr_lss_0.3", same_init_state_repeated_times, threshold=0.3)
        extract_useful_data(eval_demo_path, "sr_lss_0.7", same_init_state_repeated_times, threshold=0.7)
        extract_useful_data(eval_demo_path, "sr_lss_0.5", same_init_state_repeated_times, threshold=0.5)
        logging.info("Extracted useful data from {}_seed_{}".format(prefix, seed))

def start_generate_challenge_data(eval_path, seeds, prefix, eval_output_dir_name="eval"):
    for seed in seeds:
        eval_demo_path = os.path.join(eval_path, eval_output_dir_name, "demo.hdf5")


def start_combining_dataset(used_demo, core_dataset_path, eval_path, seeds, prefix, eval_output_dir_name="eval"):
    for seed in seeds:
        eval_demo_path = os.path.join(eval_path, eval_output_dir_name, "demo_adv.hdf5")
        output_demo_path = os.path.join(eval_path, eval_output_dir_name, "demo_plus_adv.hdf5")
        dataset_combine_cfg = EasyDict({
            "output": output_demo_path,
            "input": [
                {
                    "path": eval_demo_path,
                    "map":[
                        [None, "all"],
                        ["sr_lss_0.3", "train_0.3"],
                        ["sr_lss_0.7", "train_0.7"],
                        ["sr_lss_0.5", "train_0.5"],
                        ["success", "train_success"]
                    ]
                }, 
                {
                    "path": core_dataset_path,
                    "map":[
                        [used_demo, "all"],
                        [used_demo, "train_0.3"],
                        [used_demo, "train_0.7"],
                        [used_demo, "train_0.5"],
                        [used_demo, "train_success"]
                    ]
                }
            ]
        })
        combine_dataset(dataset_combine_cfg)
        logging.info("Combined dataset {}_seed_{} and core".format(prefix, seed))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to xxx_v141.hdf5 dataset file, xxx can be low_dim, image, etc.",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="task config name, e.g. can, lift, square, tool_hang, transport",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory for saving results",
    )

    parser.add_argument(
        "--used_demo",
        type=str,
        required=True,
        help="e.g. core_5, core_10, core_20, core_40, core_50, core_100, core_200",
    )

    parser.add_argument(
        "--seeds",
        default=[233],
        type=int,
        nargs="+",
        help="every given seed will be tested",
    )

    parser.add_argument(
        "--cuda_device",
        default=None,
        type=int,
        nargs="+",
        help="use which cuda device to train",
    )

    parser.add_argument(
        "--noise_scale",
        default=0.01,    # 0.5 for image, 0.01 for low_dim
        type=float,
        help="noise scale",
    )

    args = parser.parse_args()

    # set up logging
    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    log_file = os.path.join(abs_output_dir, "full_run.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO, filemode="w",
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ensure actions are in absolute form
    dataset_path = args.dataset
    dataset_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    assert dataset_name[-5:] == ".hdf5"
    dataset_path_of_abs_actions = os.path.join(dataset_dir, dataset_name[:-5] + "_abs_6drot.hdf5")
    if not os.path.exists(dataset_path_of_abs_actions):
        # back up the original dataset
        backup_dataset_path = os.path.join(dataset_dir, dataset_name[:-5] + ".bak.hdf5")
        os.system("cp {} {}".format(dataset_path, backup_dataset_path))
        logging.info("Backed up the original dataset to {}".format(backup_dataset_path))

        # count the number of demos
        f = h5py.File(dataset_path, "r")
        num_demos = len(list(f["data"].keys()))
        f.close()
        logging.info("Number of demos: {}".format(num_demos))
        assert num_demos == 200

        # extract demos
        assert args.used_demo in ["core_5", "core_10", "core_20", "core_40", "core_50", "core_100", "core_200"]
        extract_demo(dataset_path, "core_5", num_demos//5)
        extract_demo(dataset_path, "core_10", num_demos//10)
        extract_demo(dataset_path, "core_20", num_demos//20)
        extract_demo(dataset_path, "core_40", num_demos//40)
        extract_demo(dataset_path, "core_50", num_demos//50)
        extract_demo(dataset_path, "core_100", num_demos//100)
        extract_demo(dataset_path, "core_200", num_demos//200)
        logging.info("Extracted core_5, core_10, core_20, core_40, core_50, core_100, core_200")

        # convert relative actions to absolute actions
        eval_dir = os.path.join(dataset_dir, dataset_name[:-5] + "_abs_6drot_eval")
        convert_rel_actions_to_abs(dataset_path, dataset_path_of_abs_actions, eval_dir, num_workers=None)
        logging.info("Converted relative actions to absolute actions")
    else:
        logging.info("Absolute action dataset already exist")
    core_path = dataset_path = dataset_path_of_abs_actions
    abs_core_path = os.path.abspath(core_path)

    # generate training configs
    # generate_training_cfg(args.task, args.used_demo, abs_core_path, abs_output_dir, args.seeds, "base")
    # start training
    # start_training(abs_output_dir, args.seeds, "base", cuda_device=args.cuda_device)

    eval_output_path = os.path.join(abs_output_dir)

    # evaluate noisy demos (for sime)
    # start_evaluating(abs_output_dir, args.seeds, "base", eval_specific_args="--enable_exploration --tau1 0.0 --tau2 1.0 --noise_scale {}".format(args.noise_scale), eval_output_dir_name=eval_output_path, cuda_device=args.cuda_device)
    # generate challenge data
    start_evaluating(abs_output_dir, args.seeds, "base", eval_specific_args="--enable_exploration --tau1 0.0 --tau2 1.0 --noise_scale {}".format(args.noise_scale), eval_output_dir_name=eval_output_path, cuda_device=args.cuda_device, adv=True)

    # extract noisy data
    start_extracting_useful_data(eval_output_path, args.seeds, "base", eval_output_dir_name="")
    start_combining_dataset(args.used_demo, abs_core_path, eval_output_path, args.seeds, "sime", eval_output_dir_name="")


    # # base eval
    # start_evaluating(abs_output_dir, args.seeds, "base", cuda_device=args.cuda_device)

    # # sime train
    # sime_dataset = os.path.join(eval_output_path, "demo_plus_core.hdf5")
    sime_dataset = os.path.join(eval_output_path, "demo_plus_adv.hdf5")

    print(sime_dataset)
    generate_training_cfg(args.task, "train_0.3", sime_dataset, abs_output_dir, args.seeds, "sime")

    start_training(abs_output_dir, args.seeds, "sime", cuda_device=args.cuda_device)

    # sime eval
    start_evaluating(abs_output_dir, args.seeds, "sime", cuda_device=args.cuda_device)


    # # chase train
    # start_evaluating(abs_output_dir, seeds, "chase", eval_specific_args="--enable_adv --noise_scale {}".format(args.noise_scale), eval_output_dir_name=eval_output_dir_name, cuda_device=args.cuda_device)
    # # extract useful data
    # start_extracting_useful_data(eval_output_path, args.seeds, "chase", eval_output_dir_name="chase")
    # start_combining_dataset(args.used_demo, abs_core_path, eval_output_path, args.seeds, "chase", eval_output_dir_name="chase")

    # generate_training_cfg(args.task, args.used_demo, abs_core_path, abs_output_dir, args.seeds, "round_0")
    # start_training(abs_output_dir, args.seeds, "round_1", cuda_device=args.cuda_device)

    # # chase eval noisy
    # start_evaluating(abs_output_dir, args.seeds, "round_0", cuda_device=args.cuda_device)


main()