import math
import os

# genesis must be imported before rsl_rl / tensorboard (see g1_env.py)
import genesis as gs
import torch
from tensordict import TensorDict

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def hand_target_quat(finger_axis, palm_dir):
    """Quaternion (w, x, y, z) of a link whose local `finger_axis` (0/1/2) points to world +z and whose local unit
    vector `palm_dir` (orthogonal to it) points to world +x (forward)."""
    f = [1.0 if i == finger_axis else 0.0 for i in range(3)]
    p = [float(v) for v in palm_dir]
    t = [f[1] * p[2] - f[2] * p[1], f[2] * p[0] - f[0] * p[2], f[0] * p[1] - f[1] * p[0]]  # f x p -> world y
    # rows of R (world <- local) are the local vectors p, t, f
    r = [p, t, f]
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s]
    i = max(range(3), key=lambda k: r[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(r[i][i] - r[j][j] - r[k][k] + 1.0) * 2
    q = [0.0] * 4
    q[0] = (r[k][j] - r[j][k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (r[j][i] + r[i][j]) / s
    q[1 + k] = (r[k][i] + r[i][k]) / s
    return q


CART_XML = """<mujoco model="push_cart">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>

  <default>
    <geom friction="1.0 0.005 0.0001" condim="3"/>
    <joint damping="0.1"/>
  </default>

  <worldbody>
    <body name="cart_base" pos="0 0 0.26">
      <freejoint name="cart_free"/>
      <inertial pos="0 0 0" mass="12" diaginertia="0.65 0.9 1.3"/>

      <geom name="platform" type="box" size="0.4 0.3 0.06" rgba="0.25 0.45 0.7 1" mass="0"/>

      <!-- Handle on the rear (-X) -->
      <geom name="handle_left" type="capsule" fromto="{x} {w} 0 {x} {w} {h}" size="0.018" mass="0"/>
      <geom name="handle_right" type="capsule" fromto="{x} {nw} 0 {x} {nw} {h}" size="0.018" mass="0"/>
      <geom name="handle_bar" type="capsule" fromto="{x} {nw} {h} {x} {w} {h}" size="0.022" mass="0"/>

      <!-- Wheels rotate around local Y -->
      <body name="wheel_fl" pos="0.28 0.25 -0.16">
        <joint name="wheel_fl_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.10 0.035" euler="1.5707963 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_fr" pos="0.28 -0.25 -0.16">
        <joint name="wheel_fr_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.10 0.035" euler="1.5707963 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_rl" pos="-0.28 0.25 -0.16">
        <joint name="wheel_rl_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.10 0.035" euler="1.5707963 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_rr" pos="-0.28 -0.25 -0.16">
        <joint name="wheel_rr_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.10 0.035" euler="1.5707963 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def write_cart_xml(path, handle_height, handle_x, handle_half_width):
    """Write the cart MJCF. The handle bar is at height `handle_height` (above the cart base), x = `handle_x` and
    spans y = +-`handle_half_width` (cart frame)."""
    xml = (
        CART_XML.replace("{h}", f"{handle_height:.4f}")
        .replace("{x}", f"{handle_x:.4f}")
        .replace("{w}", f"{handle_half_width:.4f}")
        .replace("{nw}", f"{-handle_half_width:.4f}")
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(xml)
    return path


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class G1CartEnv:
    """Unitree G1 pushing a cart. The command is the target velocity of the CART (in the cart frame).

    Both wrists are welded to the cart (hands fixed to the handle). By default the policy drives legs, waist and arms
    (29 joints): with the hands fixed, the upper body is used to push/press on the handle so that the cart wheels stay
    on the ground (penalties: wheels off the ground, cart tilt). With num_actions = 12 only the legs are driven and the
    upper body holds the default pose.
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, record=False):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]  # 29 joints: legs 0-11, waist 12-14, arms 15-28
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
                max_collision_pairs=100,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -2.5, 1.5), camera_lookat=(0.5, 0.0, 0.7), camera_fov=40
            ),
            show_viewer=show_viewer,
        )
        self._add_ground()
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=env_cfg["mjcf_path"],
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            )
        )
        # the cart has a fixed shape; the robot's initial arm pose is solved (IK) to grasp its handle
        self.cart_base_z = 0.26
        self.handle_height = env_cfg["handle_height"]
        self.handle_half_width = env_cfg["handle_half_width"]
        self.handle_x = env_cfg["handle_x"]
        cart_xml = write_cart_xml(env_cfg["cart_xml_path"], self.handle_height, self.handle_x, self.handle_half_width)
        self.cart = self.scene.add_entity(gs.morphs.MJCF(file=cart_xml, pos=(1.0, 0.0, 0.26)))

        self.cam = None
        if record:
            self.cam = self.scene.add_camera(res=(960, 540), pos=(3.0, -2.5, 1.5), lookat=(0.5, 0.0, 0.7), fov=40)

        self.scene.build(n_envs=num_envs)

        # ---- joints: all 29 are PD-controlled. The policy drives all of them, or only the legs (num_actions = 12)
        # while the waist and arms hold the default pose.
        names = env_cfg["joint_names"]
        self.all_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start for n in names], dtype=gs.tc_int, device=gs.device
        )
        ctrl_cols = list(range(self.num_actions))
        assert self.num_actions in (12, 29), "num_actions must be 12 (legs only) or 29 (legs + waist + arms)"
        assert len(ctrl_cols) == self.num_actions
        self.ctrl_cols = torch.tensor(ctrl_cols, dtype=torch.long, device=gs.device)
        self.ctrl_dof_idx = self.all_dof_idx[self.ctrl_cols]
        self.robot.set_dofs_kp([env_cfg["gains"][n][0] for n in names], self.all_dof_idx)
        self.robot.set_dofs_kv([env_cfg["gains"][n][1] for n in names], self.all_dof_idx)
        fmax = torch.tensor([env_cfg["gains"][n][2] for n in names], dtype=gs.tc_float, device=gs.device)
        self.robot.set_dofs_force_range(-fmax, fmax, self.all_dof_idx)

        # ---- initial state
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.unit_x = torch.tensor([1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        def make_qpos(angles):
            init_dof = torch.tensor(
                [angles[j.name] for j in self.robot.joints[1:]], dtype=gs.tc_float, device=gs.device
            )
            return torch.concatenate((self.init_base_pos, self.init_base_quat, init_dof))

        # ---- fixed cart: its pose and the points on the handle bar where the wrists have to be
        cart_x = env_cfg["cart_init_x"]
        self.init_cart_pos = torch.tensor([cart_x, 0.0, self.cart_base_z], dtype=gs.tc_float, device=gs.device)
        self.init_cart_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
        self.init_cart_qpos = torch.concatenate(
            (self.init_cart_pos, self.init_cart_quat, torch.zeros(4, dtype=gs.tc_float, device=gs.device))
        )  # free joint + 4 wheels
        bar_centre = self.init_cart_pos + torch.tensor(
            [self.handle_x, 0.0, self.handle_height], dtype=gs.tc_float, device=gs.device
        )
        grasp_y = env_cfg["grasp_y"]
        grasp_back = env_cfg["grasp_offset"]  # wrists stay this far behind the bar centre; the palms push its rear face
        grasp_drop = env_cfg["grasp_drop"]  # wrists sit this far below the bar: the palms (above the wrist) hit the bar
        grasp_targets = [  # left (+y), right (-y), same order as hand_link_names
            bar_centre + torch.tensor([-grasp_back, grasp_y, -grasp_drop], dtype=gs.tc_float, device=gs.device),
            bar_centre + torch.tensor([-grasp_back, -grasp_y, -grasp_drop], dtype=gs.tc_float, device=gs.device),
        ]

        # ---- initial arm pose: IK on the waist and arm joints so that both wrists reach the grasp points
        hand_links = [self.robot.get_link(n) for n in env_cfg["hand_link_names"]]
        angles = dict(env_cfg["default_joint_angles"])
        self.robot.set_qpos(make_qpos(angles))
        upper_names = names[12:]  # waist + arms
        upper_dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in upper_names]
        # full wrist orientation (rot_mask all True): fingertips point to world +z and the palms face forward (+x)
        finger_axis = env_cfg["finger_axis"]  # wrist_yaw_link local axis along the fingers (0: x, 1: y, 2: z)
        palm_dirs = env_cfg["palm_dirs"]  # local palm-normal vector of each hand (left, right)
        rot_mask = [True, True, True]
        up_quats = [
            torch.tensor(hand_target_quat(finger_axis, d), dtype=gs.tc_float, device=gs.device) for d in palm_dirs
        ]
        ik_qpos, ik_err = self.robot.inverse_kinematics_multilink(
            links=hand_links,
            poss=grasp_targets,
            quats=up_quats,
            rot_mask=rot_mask,
            dofs_idx_local=upper_dofs,
            return_error=True,
            envs_idx=[0],
        )
        for n in upper_names:
            angles[n] = float(ik_qpos[0, self.robot.get_joint(n).qs_idx_local[0]])
        print(f"[G1CartEnv] IK pose error (per link: pos xyz [m], rot xyz [rad]): {ik_err[0].tolist()}")
        for name, e in zip(env_cfg["hand_link_names"], ik_err[0].tolist()):
            if max(abs(v) for v in e[:3]) > 0.03 or max(abs(v) for v in e[3:]) > 0.1:
                print(f"[G1CartEnv] WARNING: IK did not converge for {name} (pos/rot error {e}); try another palm_dirs sign")
        print("[G1CartEnv] grasp pose:", {n: round(angles[n], 3) for n in upper_names})

        # the grasp pose becomes the default pose: PD targets and the reset pose of the waist and arms
        self.init_qpos = make_qpos(angles)
        self.default_all = torch.tensor([angles[n] for n in names], dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = self.default_all[self.ctrl_cols]

        # ---- place the cart and weld both wrists to it (the weld keeps the relative pose they have right now,
        # which is the start pose of every episode)
        self.robot.set_qpos(self.init_qpos)
        self.cart.set_qpos(self.init_cart_qpos)
        hand_pos = torch.stack([l.get_pos()[0] for l in hand_links]).to(gs.tc_float)  # (2, 3)
        cart_link = self.cart.get_link("cart_base")
        rigid = self.scene.sim.rigid_solver
        for hand in hand_links:
            rigid.add_weld_constraint(cart_link.idx, hand.idx)
        hand_quats = [l.get_quat()[0].to(gs.tc_float) for l in hand_links]
        finger_vec = torch.eye(3, dtype=gs.tc_float, device=gs.device)[finger_axis]
        finger_world = [transform_by_quat(finger_vec, q).tolist() for q in hand_quats]
        palm_world = [
            transform_by_quat(torch.tensor(d, dtype=gs.tc_float, device=gs.device), q).tolist()
            for d, q in zip(palm_dirs, hand_quats)
        ]
        print(f"[G1CartEnv] fingertip direction in world (should be [0, 0, 1]): {finger_world}")
        print(f"[G1CartEnv] palm direction in world (should be [1, 0, 0]): {palm_world}")
        print(
            f"[G1CartEnv] wrist pos {hand_pos.tolist()}, targets {[t.tolist() for t in grasp_targets]}, "
            f"handle centre {bar_centre.tolist()} (half width {self.handle_half_width:.3f} m)"
        )

        # ---- cart wheels (ground contact is read from the net contact force of each wheel link)
        self.wheel_idx_local = [self.cart.get_link(n).idx_local for n in env_cfg["wheel_link_names"]]

        # ---- feet
        self.feet_links = [self.robot.get_link(n) for n in env_cfg["feet_link_names"]]
        self.feet_idx_local = [l.idx_local for l in self.feet_links]
        self.feet_z_stand = torch.stack([l.get_pos()[0, 2] for l in self.feet_links]).to(gs.tc_float)

        # ---- buffers
        f = dict(dtype=gs.tc_float, device=gs.device)
        n = num_envs
        self.base_lin_vel = torch.zeros((n, 3), **f)
        self.base_ang_vel = torch.zeros((n, 3), **f)
        self.projected_gravity = torch.zeros((n, 3), **f)
        self.rew_buf = torch.zeros((n,), **f)
        self.reset_buf = torch.ones((n,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.zeros((n, self.num_commands), **f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], **f
        )
        self.commands_limits = tuple(
            torch.tensor(v, **f)
            for v in zip(
                command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"]
            )
        )
        self.actions = torch.zeros((n, self.num_actions), **f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((n, 3), **f)
        self.base_quat = torch.zeros((n, 4), **f)
        self.cart_pos = torch.zeros((n, 3), **f)
        self.cart_quat = torch.zeros((n, 4), **f)
        self.cart_lin_vel = torch.zeros((n, 3), **f)  # cart frame
        self.cart_ang_vel = torch.zeros((n, 3), **f)  # world frame
        self.wheel_contact = torch.ones((n, len(self.wheel_idx_local)), dtype=gs.tc_bool, device=gs.device)
        self.feet_z = torch.zeros((n, 2), **f)
        self.feet_contact = torch.zeros((n, 2), dtype=gs.tc_bool, device=gs.device)
        self.phase = torch.zeros((n,), **f)
        self.leg_phase = torch.zeros((n, 2), **f)
        self.extras = dict()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((n,), **f)

        self.upper_idx = list(range(12, self.num_actions))  # waist + arms (empty when only the legs are driven)
        self.hip_ry_idx = [i for i, nm in enumerate(names[:12]) if "hip_roll" in nm or "hip_yaw" in nm]

        self.reset()

    # ------------------------------------------------------------------ helpers
    def _add_ground(self):
        """Flat ground. Subclasses override this (and _ground_height) to use another terrain."""
        self.scene.add_entity(gs.morphs.Plane())

    def _ground_height(self, xy):
        """Terrain height [m] below the world positions xy (n, 2)."""
        return torch.zeros((xy.shape[0],), dtype=gs.tc_float, device=gs.device)

    def _cart_projected_gravity(self):
        return transform_by_quat(self.global_gravity, inv_quat(self.cart_quat))

    def _cart_forward_in_base(self):
        fwd = transform_by_quat(self.unit_x, self.cart_quat)
        return transform_by_quat(fwd, inv_quat(self.base_quat))

    # ------------------------------------------------------------------ simulation
    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
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
        target[:, self.ctrl_cols] += exec_actions * self.env_cfg["action_scale"]
        self.robot.control_dofs_position(target, self.all_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_height = self.base_pos[:, 2] - self._ground_height(self.base_pos[:, :2])  # above the terrain
        self.base_quat = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.ctrl_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.ctrl_dof_idx)

        self.cart_pos = self.cart.get_pos()
        self.cart_quat = self.cart.get_quat()
        self.cart_lin_vel = transform_by_quat(self.cart.get_vel(), inv_quat(self.cart_quat))
        self.cart_ang_vel = self.cart.get_ang()
        cart_forces = self.cart.get_links_net_contact_force()
        self.wheel_contact = torch.norm(cart_forces[:, self.wheel_idx_local], dim=-1) > self.env_cfg["wheel_contact_force"]

        feet_pos = [l.get_pos() for l in self.feet_links]
        self.feet_z = torch.stack([p[:, 2] - self._ground_height(p[:, :2]) for p in feet_pos], dim=1)  # above terrain
        forces = self.robot.get_links_net_contact_force()
        self.feet_contact = torch.norm(forces[:, self.feet_idx_local], dim=-1) > 1.0

        self.phase = (self.episode_length_buf * self.dt / self.env_cfg["gait_period"]) % 1.0
        self.leg_phase = torch.stack([self.phase, (self.phase + 0.5) % 1.0], dim=1)

        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # termination: robot falls
        timeout = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timeout.clone()
        self.reset_buf |= self.base_height < self.env_cfg["termination_height"]
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
        self.cart.set_qpos(self.init_cart_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            self.cart_pos.copy_(self.init_cart_pos)
            self.cart_quat.copy_(self.init_cart_quat)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.cart_lin_vel.zero_()
            self.cart_ang_vel.zero_()
            self.wheel_contact.fill_(True)
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
            torch.where(m, self.init_cart_pos, self.cart_pos, out=self.cart_pos)
            torch.where(m, self.init_cart_quat, self.cart_quat, out=self.cart_quat)
            self.base_lin_vel.masked_fill_(m, 0.0)
            self.base_ang_vel.masked_fill_(m, 0.0)
            self.cart_lin_vel.masked_fill_(m, 0.0)
            self.cart_ang_vel.masked_fill_(m, 0.0)
            self.wheel_contact.masked_fill_(m, True)
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
        inv_base_quat = inv_quat(self.base_quat)
        cart_rel = transform_by_quat(self.cart_pos - self.base_pos, inv_base_quat)  # 3
        cart_fwd = self._cart_forward_in_base()[:, :2]  # 2
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3 (cart velocity command)
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 29
                self.dof_vel * self.obs_scales["dof_vel"],  # 29
                self.actions,  # 29
                torch.sin(2 * math.pi * self.phase)[:, None],  # 1
                torch.cos(2 * math.pi * self.phase)[:, None],  # 1
                cart_rel,  # 3
                self.cart_lin_vel * self.obs_scales["lin_vel"],  # 3
                cart_fwd,  # 2
                self._cart_projected_gravity(),  # 3 (cart tilt)
                self.wheel_contact.float(),  # 4 (wheel-ground contact)
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ------------------------------------------------------------------ rewards
    def _reward_tracking_cart_lin_vel(self):
        err = torch.sum(torch.square(self.commands[:, :2] - self.cart_lin_vel[:, :2]), dim=1)
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_cart_ang_vel(self):
        err = torch.square(self.commands[:, 2] - self.cart_ang_vel[:, 2])
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_wheels_off_ground(self):
        # number of cart wheels that are not touching the ground
        return torch.sum((~self.wheel_contact).float(), dim=1)

    def _reward_cart_tilt(self):
        # cart roll / pitch (a wheelie or a sideways tilt shows up in the projected gravity)
        return torch.sum(torch.square(self._cart_projected_gravity()[:, :2]), dim=1)

    def _reward_upper_body_pos(self):
        # mild regularisation so that waist/arm motions stay reasonable
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.upper_idx], dim=1)

    def _reward_heading(self):
        fwd = self._cart_forward_in_base()[:, :2]
        return 1.0 - fwd[:, 0] / torch.norm(fwd, dim=1).clamp(min=1e-6)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_height - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_ry_idx], dim=1)

    def _reward_contact(self):
        stance = self.leg_phase < 0.55
        return torch.sum((self.feet_contact == stance).float(), dim=1)

    def _reward_feet_swing_height(self):
        swing = self.leg_phase >= 0.55
        target = self.feet_z_stand[None, :] + self.reward_cfg["feet_height_target"]
        return torch.sum(torch.square(self.feet_z - target) * swing, dim=1)

    def _reward_alive(self):
        return torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)
