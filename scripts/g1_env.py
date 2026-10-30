import math

# genesis must be imported before rsl_rl / tensorboard (importing those first makes the EGL init segfault on Colab)
import genesis as gs
import torch
from tensordict import TensorDict

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class G1Env:
    """Unitree G1 velocity-tracking environment (legs are learned, waist/arms hold the default pose)."""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, record=False):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.02  # control frequency 50 Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = dict(reward_cfg["reward_scales"])

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=40,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 0.0, 1.5), camera_lookat=(0.0, 0.0, 0.7), camera_fov=40
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=env_cfg["mjcf_path"],
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            )
        )
        self.cam = None
        if record:
            self.cam = self.scene.add_camera(res=(960, 540), pos=(2.5, -2.0, 1.4), lookat=(0.0, 0.0, 0.7), fov=40)

        self.scene.build(n_envs=num_envs)

        # ---- joints: all actuated joints are PD-controlled, only the legs are driven by the policy
        names = env_cfg["joint_names"]  # 29 actuators in MJCF order, first 12 are the legs
        self.all_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start for n in names], dtype=gs.tc_int, device=gs.device
        )
        self.leg_dof_idx = self.all_dof_idx[: self.num_actions]
        kp = [env_cfg["gains"][n][0] for n in names]
        kd = [env_cfg["gains"][n][1] for n in names]
        fmax = [env_cfg["gains"][n][2] for n in names]
        self.robot.set_dofs_kp(kp, self.all_dof_idx)
        self.robot.set_dofs_kv(kd, self.all_dof_idx)
        fmax = torch.tensor(fmax, dtype=gs.tc_float, device=gs.device)
        self.robot.set_dofs_force_range(-fmax, fmax, self.all_dof_idx)

        self.default_all = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in names], dtype=gs.tc_float, device=gs.device
        )
        self.default_dof_pos = self.default_all[: self.num_actions]

        # ---- initial state (qpos order follows robot.joints[1:])
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        init_dof = torch.tensor(
            [env_cfg["default_joint_angles"][j.name] for j in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, init_dof))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # ---- feet
        self.feet_links = [self.robot.get_link(n) for n in env_cfg["feet_link_names"]]
        self.feet_idx_local = [l.idx_local for l in self.feet_links]
        self.robot.set_qpos(self.init_qpos)
        self.feet_z_stand = torch.stack([l.get_pos()[0, 2] for l in self.feet_links]).to(gs.tc_float)  # (2,)

        # ---- buffers
        f = dict(dtype=gs.tc_float, device=gs.device)
        self.base_lin_vel = torch.zeros((num_envs, 3), **f)
        self.base_ang_vel = torch.zeros((num_envs, 3), **f)
        self.projected_gravity = torch.zeros((num_envs, 3), **f)
        self.rew_buf = torch.zeros((num_envs,), **f)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.zeros((num_envs, self.num_commands), **f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], **f
        )
        self.commands_limits = tuple(
            torch.tensor(v, **f)
            for v in zip(
                command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"]
            )
        )
        self.actions = torch.zeros((num_envs, self.num_actions), **f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((num_envs, 3), **f)
        self.base_quat = torch.zeros((num_envs, 4), **f)
        self.base_euler = torch.zeros((num_envs, 3), **f)
        self.feet_z = torch.zeros((num_envs, 2), **f)
        self.feet_contact = torch.zeros((num_envs, 2), dtype=gs.tc_bool, device=gs.device)
        self.phase = torch.zeros((num_envs,), **f)
        self.leg_phase = torch.zeros((num_envs, 2), **f)
        self.extras = dict()

        # reward functions (scales multiplied by dt)
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((num_envs,), **f)

        # hip roll / yaw joints (indices inside the 12 leg joints)
        self.hip_ry_idx = [i for i, n in enumerate(names[: self.num_actions]) if "hip_roll" in n or "hip_yaw" in n]

        self.reset()

    # ------------------------------------------------------------------
    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        # sometimes ask the robot to stand still
        still = torch.rand((self.num_envs,), device=gs.device) < self.command_cfg["still_prob"]
        commands = torch.where(still[:, None], torch.zeros_like(commands), commands)
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.env_cfg["simulate_action_latency"] else self.actions
        target = self.default_all.repeat(self.num_envs, 1)
        target[:, : self.num_actions] += exec_actions * self.env_cfg["action_scale"]
        self.robot.control_dofs_position(target, self.all_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.leg_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.leg_dof_idx)

        self.feet_z = torch.stack([l.get_pos()[:, 2] for l in self.feet_links], dim=1)
        forces = self.robot.get_links_net_contact_force()
        self.feet_contact = torch.norm(forces[:, self.feet_idx_local], dim=-1) > 1.0

        # gait clock (also part of the observation)
        self.phase = (self.episode_length_buf * self.dt / self.env_cfg["gait_period"]) % 1.0
        self.leg_phase = torch.stack([self.phase, (self.phase + 0.5) % 1.0], dim=1)

        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # termination
        timeout = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timeout.clone()
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_height"]
        self.reset_buf |= self.projected_gravity[:, 2] > self.env_cfg["termination_gravity_z"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = timeout.to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            m = envs_idx[:, None]
            torch.where(m, self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(m, self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(m, self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(m, self.default_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(m, 0.0)
            self.base_ang_vel.masked_fill_(m, 0.0)
            self.dof_vel.masked_fill_(m, 0.0)
            self.actions.masked_fill_(m, 0.0)
            self.last_actions.masked_fill_(m, 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                torch.sin(2 * math.pi * self.phase)[:, None],  # 1
                torch.cos(2 * math.pi * self.phase)[:, None],  # 1
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ------------------------------------------------------------------ rewards
    def _reward_tracking_lin_vel(self):
        err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_ry_idx], dim=1)

    def _reward_contact(self):
        # stance during the first 55% of each leg's phase, swing afterwards
        stance = self.leg_phase < 0.55
        return torch.sum((self.feet_contact == stance).float(), dim=1)

    def _reward_feet_swing_height(self):
        swing = self.leg_phase >= 0.55
        target = self.feet_z_stand[None, :] + self.reward_cfg["feet_height_target"]
        return torch.sum(torch.square(self.feet_z - target) * swing, dim=1)

    def _reward_alive(self):
        return torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)
