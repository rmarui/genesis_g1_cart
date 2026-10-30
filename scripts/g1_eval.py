"""Evaluate velocity tracking of a trained G1 policy, optionally recording a video.

    python g1_eval.py --vx 0.5                      # numbers only
    python g1_eval.py --vx 0.5 --record             # fixed command, video saved to g1_eval.mp4
    python g1_eval.py --vx 0.5 --video g1.mp4       # same, with a chosen file name
    python g1_eval.py --demo --video g1_demo.mp4    # several commands in a row, video
"""
import argparse
import pickle

# genesis must be imported before rsl_rl / tensorboard (see g1_env.py)
import genesis as gs

from g1_env import G1Env
from g1_train import get_cfgs

import torch
from rsl_rl.runners import OnPolicyRunner

# (duration [s], vx, vy, wz) used by --demo
DEMO_SCHEDULE = [
    (3.0, 0.0, 0.0, 0.0),
    (4.0, 0.5, 0.0, 0.0),
    (4.0, 0.8, 0.0, 0.0),
    (4.0, 0.4, 0.0, 0.6),
    (4.0, 0.4, 0.0, -0.6),
    (3.0, 0.0, 0.3, 0.0),
    (3.0, 0.0, 0.0, 0.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-walking")
    parser.add_argument("--log-root", type=str, default="logs", help="e.g. /content/drive/MyDrive/g1_logs")
    parser.add_argument("--ckpt", type=int, default=-1, help="checkpoint iteration (-1: latest)")
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--demo", action="store_true", help="run DEMO_SCHEDULE instead of a fixed command")
    parser.add_argument("--record", action="store_true", help="record a video (G1Env record=True)")
    parser.add_argument("--video", type=str, default=None, help="output mp4 path (implies --record)")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.record and args.video is None:
        args.video = "g1_eval.mp4"

    log_dir = f"{args.log_root}/{args.exp_name}"
    saved = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(saved["mjcf"])
    reward_cfg["reward_scales"] = {}
    command_cfg["still_prob"] = 0.0
    env_cfg["episode_length_s"] = 1000.0

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env = G1Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=args.viewer, record=args.video is not None)

    runner = OnPolicyRunner(env, saved["train_cfg"], None, device=gs.device)  # None: no TensorBoard writer
    import glob, os, re

    if args.ckpt >= 0:
        path = f"{log_dir}/model_{args.ckpt}.pt"
    else:
        cks = glob.glob(f"{log_dir}/model_*.pt")
        path = max(cks, key=lambda p: int(re.findall(r"model_(\d+)", os.path.basename(p))[0]))
    print("loading", path)
    runner.load(path)
    policy = runner.get_inference_policy(device=gs.device)

    if args.demo:
        schedule = DEMO_SCHEDULE
    else:
        schedule = [(args.steps * env.dt, args.vx, args.vy, args.wz)]

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
                    torch.cat([env.base_lin_vel[0, :2] - cmd[:2], (env.base_ang_vel[0, 2] - cmd[2])[None]])
                    .abs()
                    .cpu()
                )
                falls += int(done[0])
                if args.video is not None:
                    p = env.base_pos[0].cpu().numpy()
                    env.cam.set_pose(pos=(p[0] + 2.5, p[1] - 2.0, 1.4), lookat=(p[0], p[1], 0.7))

    if args.video is not None:
        env.cam.stop_recording()
        print("saved", args.video)

    errs = torch.stack(errs)
    if not args.demo:
        print(f"command (vx, vy, wz) = {[args.vx, args.vy, args.wz]}")
    print("mean |error| (vx, vy, wz):", errs.mean(0).tolist())
    print("resets during eval:", falls)


if __name__ == "__main__":
    main()
