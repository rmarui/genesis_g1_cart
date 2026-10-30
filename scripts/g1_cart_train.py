import argparse
import glob
import os
import pickle
import re
import shutil
import time
from importlib import metadata

# genesis must be imported before rsl_rl / tensorboard (see g1_env.py)
import genesis as gs

from g1_cart_env import G1CartEnv
from g1_train import JOINT_NAMES, STAND_CTRL, get_train_cfg, pd_gains

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner


def get_cfgs(mjcf_path, upper_body=True):
    """upper_body=True: the policy also drives waist + arms (29 actions); False: legs only (12 actions)."""
    env_cfg = {
        "mjcf_path": mjcf_path,
        "cart_xml_path": "assets/cart.xml",
        "num_actions": 29 if upper_body else 12,  # legs 12 (+ waist 3 + arms 14); the hands are welded to the cart
        "joint_names": JOINT_NAMES,
        "default_joint_angles": dict(zip(JOINT_NAMES, STAND_CTRL)),
        "gains": {n: pd_gains(n) for n in JOINT_NAMES},  # name -> (kp, kd, max torque)
        "feet_link_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "hand_link_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "wheel_link_names": ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"],
        "wheel_contact_force": 1.0,  # [N] a wheel counts as touching the ground above this net contact force
        # Fixed cart shape: the handle bar is at x = handle_x and height handle_height above the cart base (cart frame,
        # base z = 0.26 m, so the bar is 0.85 m above the ground) and spans y = +-handle_half_width.
        "handle_x": -0.34,
        "handle_height": 0.59,
        "handle_half_width": 0.22,
        # Fixed cart start: x of the cart base. The robot's waist/arms are solved with IK so that the wrists are at
        # y = +-grasp_y on the bar, grasp_offset [m] behind its centre (palms push the rear face).
        "cart_init_x": 0.64,
        "grasp_y": 0.20,
        "grasp_offset": 0.05,
        # wrist_yaw_link local axis pointing along the fingers (0: x, 1: y, 2: z); IK makes it point to world +z.
        # Check the "fingertip direction" line in the startup log; if it is wrong, try another axis.
        "finger_axis": 0,
        # wrist_yaw_link local unit vector of the palm normal (left hand, right hand); IK makes it point forward (+x).
        # Must be orthogonal to the finger axis. If the palms end up facing backwards in the video, flip the signs.
        "palm_dirs": [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],  # the hand meshes are mirrored, so the signs differ
        # the wrist target is this far below the bar centre (the palm lies between wrist and bar)
        "grasp_drop": 0.05,
        "base_init_pos": [0.0, 0.0, 0.79],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "termination_height": 0.5,
        "termination_gravity_z": -0.7,
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
            "tracking_cart_lin_vel": 3.0,
            "tracking_cart_ang_vel": 1.0,
            "heading": -1.0,
            "wheels_off_ground": -2.0,  # per wheel that lifts off the ground
            "cart_tilt": -5.0,
            "upper_body_pos": -0.05,
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
    # cart velocity command (cart frame): push forward, gentle steering
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.6],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [-0.3, 0.3],
        "still_prob": 0.1,
    }
    if not upper_body:
        del reward_cfg["reward_scales"]["upper_body_pos"]  # the upper body is not driven by the policy
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default=None, help="default: g1-cart, or g1-cart-legs with --fix-upper-body"
    )
    parser.add_argument(
        "--fix-upper-body",
        action="store_true",
        help="train without moving the upper body: only the 12 leg joints are driven, waist/arms hold the default pose",
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=3000)
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
    upper_body = not args.fix_upper_body

    if args.resume:
        if args.exp_name is None:  # the run is identified by its folder, so find which one holds the checkpoints
            args.exp_name = "g1-cart-legs" if args.fix_upper_body else "g1-cart"
        log_dir = f"{args.log_root}/{args.exp_name}"
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            upper_body = pickle.load(f).get("upper_body", True)  # must match the saved network (29 or 12 actions)
        if args.ckpt >= 0:
            resume_path = f"{log_dir}/model_{args.ckpt}.pt"
        else:
            cks = glob.glob(f"{log_dir}/model_*.pt")
            if not cks:
                raise FileNotFoundError(f"no model_*.pt in {log_dir}")
            resume_path = max(cks, key=lambda p: int(re.findall(r"model_(\d+)", os.path.basename(p))[0]))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(resume_path)
        print(f"resuming from {resume_path} (upper_body={upper_body})")
    else:
        if args.exp_name is None:
            args.exp_name = "g1-cart" if upper_body else "g1-cart-legs"
        log_dir = f"{args.log_root}/{args.exp_name}"  # checkpoints, cfgs.pkl (may be on Google Drive)

    tb_dir = f"{args.tb_root}/{args.exp_name}"  # TensorBoard events (local only)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.mjcf, upper_body)
    train_cfg = get_train_cfg(args.exp_name)

    if not args.resume:
        if os.path.exists(log_dir):  # never delete earlier runs (the log root may be on Google Drive)
            backup = f"{log_dir}_old_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(log_dir, backup)
            print("moved previous logs to", backup)
        os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
    os.makedirs(tb_dir, exist_ok=True)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = G1CartEnv(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)

    if not args.resume:  # on resume the existing cfgs.pkl is kept
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump({"mjcf": args.mjcf, "train_cfg": train_cfg, "upper_body": upper_body}, f)

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
# Colab:  !python -u g1_cart_train.py -b 4096 --max-iterations 3000                     (upper body driven)
#         !python -u g1_cart_train.py -b 4096 --max-iterations 3000 --fix-upper-body    (legs only)
#         !python -u g1_cart_train.py --resume --max-iterations 1000                    (1000 more iterations from the latest model_*.pt)
"""
