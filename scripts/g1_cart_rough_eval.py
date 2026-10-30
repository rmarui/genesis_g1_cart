"""Evaluate a G1 + cart policy on uneven ground: cart velocity tracking, optional video.

    python g1_cart_rough_eval.py --vx 0.4
    python g1_cart_rough_eval.py --vx 0.4 --record
    python g1_cart_rough_eval.py --demo --video cart_rough_demo.mp4
    python g1_cart_rough_eval.py --vx 0.4 --bump-height 0.03 --terrain-seed 5    # a different / rougher terrain
"""
import argparse
import glob
import os
import pickle
import re

# genesis must be imported before rsl_rl / tensorboard (see g1_env.py)
import genesis as gs

from g1_cart_eval import DEMO_SCHEDULE
from g1_cart_rough_env import G1CartRoughEnv
from g1_cart_rough_train import get_cfgs

import torch
from rsl_rl.runners import OnPolicyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-cart-rough", help="use g1-cart-rough-legs for --fix-upper-body runs")
    parser.add_argument("--log-root", type=str, default="logs", help="e.g. /content/drive/MyDrive/g1_logs")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--bump-height", type=float, default=None, help="terrain bump amplitude [m] (default: as trained)")
    parser.add_argument("--terrain-seed", type=int, default=None, help="terrain seed (default: as trained)")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--record", action="store_true", help="record a video (saved to g1_cart_rough_eval.mp4)")
    parser.add_argument("--video", type=str, default=None, help="output mp4 path (implies --record)")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.record and args.video is None:
        args.video = "g1_cart_rough_eval.mp4"

    log_dir = f"{args.log_root}/{args.exp_name}"
    saved = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    bump = args.bump_height if args.bump_height is not None else saved.get("bump_height", 0.02)
    tseed = args.terrain_seed if args.terrain_seed is not None else saved.get("terrain_seed", 0)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(saved["mjcf"], saved.get("upper_body", True), bump, tseed)
    reward_cfg["reward_scales"] = {}
    command_cfg["still_prob"] = 0.0
    env_cfg["episode_length_s"] = 1000.0

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env = G1CartRoughEnv(
        1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=args.viewer, record=args.video is not None
    )

    runner = OnPolicyRunner(env, saved["train_cfg"], None, device=gs.device)  # None: no TensorBoard writer
    if args.ckpt >= 0:
        path = f"{log_dir}/model_{args.ckpt}.pt"
    else:
        cks = glob.glob(f"{log_dir}/model_*.pt")
        path = max(cks, key=lambda p: int(re.findall(r"model_(\d+)", os.path.basename(p))[0]))
    print("loading", path)
    runner.load(path)
    policy = runner.get_inference_policy(device=gs.device)

    schedule = DEMO_SCHEDULE if args.demo else [(args.steps * env.dt, args.vx, args.vy, args.wz)]

    if args.video is not None:
        env.cam.start_recording(save_to_filename=args.video, fps=round(1.0 / env.dt))

    obs = env.get_observations()
    errs, falls = [], 0
    with torch.no_grad():
        for duration, vx, vy, wz in schedule:
            cmd = torch.tensor([vx, vy, wz], device=gs.device)
            for _ in range(int(round(duration / env.dt))):
                env.commands[:] = cmd
                obs = env.get_observations()
                obs, _, done, _ = env.step(policy(obs))
                errs.append(
                    torch.cat([env.cart_lin_vel[0, :2] - cmd[:2], (env.cart_ang_vel[0, 2] - cmd[2])[None]]).abs().cpu()
                )
                falls += int(done[0])
                if args.video is not None:
                    mid = 0.5 * (env.base_pos[0] + env.cart_pos[0]).cpu().numpy()
                    env.cam.set_pose(pos=(mid[0] + 2.5, mid[1] - 2.5, mid[2] + 1.0), lookat=(mid[0], mid[1], mid[2]))

    if args.video is not None:
        env.cam.stop_recording()
        print("saved", args.video)

    errs = torch.stack(errs)
    if not args.demo:
        print(f"cart command (vx, vy, wz) = {[args.vx, args.vy, args.wz]}")
    print(f"terrain: bump_height = {bump}, seed = {tseed}")
    print("mean |cart velocity error| (vx, vy, wz):", errs.mean(0).tolist())
    print("resets during eval:", falls)
    print(f"cart travelled to x = {env.cart_pos[0, 0].item():.2f} m (the terrain ends at x = {env_cfg['terrain']['x_range'][1]} m)")


if __name__ == "__main__":
    main()
