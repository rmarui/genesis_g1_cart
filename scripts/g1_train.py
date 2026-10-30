import argparse
import glob
import os
import pickle
import re
import shutil
import time
from importlib import metadata

# genesis must be imported before rsl_rl / tensorboard (importing those first makes the EGL init segfault on Colab)
import genesis as gs

from g1_env import G1Env

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# 'stand' keyframe ctrl of mujoco_menagerie/unitree_g1 (same order as JOINT_NAMES)
STAND_CTRL = [0.0] * 15 + [0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0] + [0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0]


def pd_gains(name):
    """(kp, kd, max torque) – same values as the notebook."""
    if "hip" in name or "knee" in name:
        return 100.0, 5.0, 88.0
    if "ankle" in name:
        return 35.0, 2.0, 40.0
    if "waist" in name:
        return 100.0, 5.0, 50.0
    if "shoulder" in name:
        return 40.0, 2.0, 25.0
    return 25.0, 1.5, 15.0


def get_train_cfg(exp_name):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs(mjcf_path):
    env_cfg = {
        "mjcf_path": mjcf_path,
        "num_actions": 12,  # leg joints; waist/arms hold the default pose
        "joint_names": JOINT_NAMES,
        "default_joint_angles": dict(zip(JOINT_NAMES, STAND_CTRL)),
        "gains": {n: pd_gains(n) for n in JOINT_NAMES},  # name -> (kp, kd, max torque)
        "feet_link_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "base_init_pos": [0.0, 0.0, 0.79],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "termination_height": 0.5,
        "termination_gravity_z": -0.7,  # projected gravity z; > -0.7 means tilted by more than ~45 deg
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "gait_period": 0.8,
    }
    obs_cfg = {"obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}}
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.74,
        "feet_height_target": 0.08,
        "reward_scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -2.0,
            "base_height": -10.0,
            "action_rate": -0.01,
            "hip_pos": -1.0,
            "contact": 0.5,
            "feet_swing_height": -20.0,
            "alive": 0.2,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.8],
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [-0.8, 0.8],
        "still_prob": 0.1,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-walking")
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mjcf", type=str, default="/content/mujoco_menagerie/unitree_g1/g1.xml")
    parser.add_argument(
        "--log-root", type=str, default="logs", help="checkpoints + cfgs.pkl go here, e.g. /content/drive/MyDrive/g1_logs"
    )
    parser.add_argument(
        "--tb-root", type=str, default="/content/tb_logs", help="TensorBoard event files go here (keep it local)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue training from a checkpoint in <log-root>/<exp-name> (default: the latest); "
        "--max-iterations then means additional iterations",
    )
    parser.add_argument("--ckpt", type=int, default=-1, help="with --resume: checkpoint iteration (-1: latest)")
    args = parser.parse_args()

    log_dir = f"{args.log_root}/{args.exp_name}"  # checkpoints, cfgs.pkl (may be on Google Drive)
    tb_dir = f"{args.tb_root}/{args.exp_name}"  # TensorBoard events (local only)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.mjcf)
    train_cfg = get_train_cfg(args.exp_name)

    if args.resume:
        if args.ckpt >= 0:
            resume_path = f"{log_dir}/model_{args.ckpt}.pt"
        else:
            cks = glob.glob(f"{log_dir}/model_*.pt")
            if not cks:
                raise FileNotFoundError(f"no model_*.pt in {log_dir}")
            resume_path = max(cks, key=lambda p: int(re.findall(r"model_(\d+)", os.path.basename(p))[0]))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(resume_path)
        print("resuming from", resume_path)
    else:
        if os.path.exists(log_dir):  # never delete earlier runs (the log root may be on Google Drive)
            backup = f"{log_dir}_old_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(log_dir, backup)
            print("moved previous logs to", backup)
        os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
    os.makedirs(tb_dir, exist_ok=True)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = G1Env(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)

    if not args.resume:  # on resume the existing cfgs.pkl is kept
        # eval re-creates the env/reward cfgs from get_cfgs(), so only the mjcf path and train cfg are needed
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump({"mjcf": args.mjcf, "train_cfg": train_cfg}, f)

    # the runner writes TensorBoard events and checkpoints into tb_dir; copy every checkpoint to log_dir as well
    runner = OnPolicyRunner(env, train_cfg, tb_dir, device=gs.device)
    if args.resume:
        runner.load(resume_path)  # restores networks, optimizer and the iteration counter

    def copy_checkpoint(path):
        if path and os.path.isfile(path) and os.path.abspath(os.path.dirname(path)) != os.path.abspath(log_dir):
            shutil.copy(path, log_dir)

    orig_save = runner.save

    def save(path, *a, **k):
        out = orig_save(path, *a, **k)
        copy_checkpoint(path)
        return out

    runner.save = save
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    for p in glob.glob(f"{tb_dir}/model_*.pt"):  # in case a checkpoint was written without going through save()
        copy_checkpoint(p)


if __name__ == "__main__":
    main()

"""
# Colab:  !python g1_train.py -b 4096 --max-iterations 1500
# tensorboard --logdir logs
"""
