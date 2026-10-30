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

from g1_cart_rough_env import G1CartRoughEnv
from g1_cart_train import get_cfgs as get_cart_cfgs
from g1_train import get_train_cfg

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner


def get_cfgs(mjcf_path, upper_body=True, bump_height=0.02, terrain_seed=0):
    """Same as the flat cart task (g1_cart_train.get_cfgs) plus an uneven-terrain config."""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cart_cfgs(mjcf_path, upper_body)
    env_cfg["terrain"] = {
        "x_range": (-2.0, 20.0),  # [m] the cart is pushed along +x; the height field ends at x = 20 m
        "y_range": (-6.0, 6.0),
        "horizontal_scale": 0.1,  # [m] grid cell
        "vertical_scale": 0.005,  # [m] height resolution
        "bump_height": bump_height,  # [m] max deviation from the flat ground (peak-to-peak is twice as much)
        "bump_wavelength": 0.25,  # [m] length scale of the bumps
        "flat_until": 1.6,  # [m] flat start area (robot + cart), the bumps fade in over `ramp` metres after it
        "ramp": 1.0,
        "seed": terrain_seed,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def latest_ckpt(log_dir):
    cks = glob.glob(f"{log_dir}/model_*.pt")
    if not cks:
        raise FileNotFoundError(f"no model_*.pt in {log_dir}")
    return max(cks, key=lambda p: int(re.findall(r"model_(\d+)", os.path.basename(p))[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default=None, help="default: g1-cart-rough, or g1-cart-rough-legs with --fix-upper-body"
    )
    parser.add_argument("--fix-upper-body", action="store_true", help="drive only the 12 leg joints (waist/arms hold the pose)")
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bump-height", type=float, default=0.02, help="terrain bump amplitude [m]")
    parser.add_argument("--terrain-seed", type=int, default=0)
    parser.add_argument("--mjcf", type=str, default="/content/mujoco_menagerie/unitree_g1/g1.xml")
    parser.add_argument("--log-root", type=str, default="logs", help="checkpoints + cfgs.pkl go here, e.g. /content/drive/MyDrive/g1_logs")
    parser.add_argument("--tb-root", type=str, default="/content/tb_logs", help="TensorBoard event files go here (keep it local)")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="start from the weights of a flat-ground cart checkpoint (e.g. <log-root>/g1-cart/model_3000.pt); "
        "the iteration counter starts from 0. The upper-body setting must match (use --fix-upper-body for g1-cart-legs).",
    )
    parser.add_argument("--resume", action="store_true", help="continue this run from its latest (or --ckpt) checkpoint")
    parser.add_argument("--ckpt", type=int, default=-1, help="with --resume: checkpoint iteration (-1: latest)")
    args = parser.parse_args()
    upper_body = not args.fix_upper_body
    if args.exp_name is None:
        args.exp_name = "g1-cart-rough" if upper_body else "g1-cart-rough-legs"

    log_dir = f"{args.log_root}/{args.exp_name}"  # checkpoints, cfgs.pkl (may be on Google Drive)
    tb_dir = f"{args.tb_root}/{args.exp_name}"  # TensorBoard events (local only)

    resume_path = None
    if args.resume:
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            saved = pickle.load(f)
        upper_body = saved.get("upper_body", True)  # must match the saved network (29 or 12 actions)
        args.bump_height = saved.get("bump_height", args.bump_height)
        args.terrain_seed = saved.get("terrain_seed", args.terrain_seed)
        resume_path = f"{log_dir}/model_{args.ckpt}.pt" if args.ckpt >= 0 else latest_ckpt(log_dir)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(resume_path)
        print(f"resuming from {resume_path} (upper_body={upper_body}, bump_height={args.bump_height})")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.mjcf, upper_body, args.bump_height, args.terrain_seed)
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

    env = G1CartRoughEnv(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)

    if not args.resume:  # on resume the existing cfgs.pkl is kept
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump(
                {
                    "mjcf": args.mjcf,
                    "train_cfg": train_cfg,
                    "upper_body": upper_body,
                    "bump_height": args.bump_height,
                    "terrain_seed": args.terrain_seed,
                },
                f,
            )

    # the runner writes TensorBoard events and checkpoints into tb_dir; copy every checkpoint to log_dir as well
    runner = OnPolicyRunner(env, train_cfg, tb_dir, device=gs.device)
    if args.resume:
        runner.load(resume_path)  # restores networks, optimizer and the iteration counter
    elif args.init_from is not None:
        print("initialising from", args.init_from)
        runner.load(args.init_from)
        runner.current_learning_iteration = 0  # a new run: count iterations (and name checkpoints) from 0

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
# Colab:  !python -u g1_cart_rough_train.py --log-root {LOG_ROOT} -b 4096 --max-iterations 3000
#         warm start from the flat-ground cart policy:
#         !python -u g1_cart_rough_train.py --log-root {LOG_ROOT} --init-from {LOG_ROOT}/g1-cart/model_3000.pt
#         !python -u g1_cart_rough_train.py --log-root {LOG_ROOT} --resume --max-iterations 1000
"""
