import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Traning for wheeled quadruped robot."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import os
from math import asin

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg, CameraCfg

import mdp

# robot model config
MOBILITY_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.environ['HOME'] + "/Documents/robot_model/param_fix_mobility.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=12.0,
            max_angular_velocity=20000.0,
            max_depenetration_velocity=None,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.6, 0.0, 0.0),
        # orientation<-(0, 0, -1.57)
        rot=(-0.7071, 0, 0, 0.7071),
        joint_pos={"caster_yaw_joint": 0.0},
    ),
    actuators={
        "left_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint"],
            effort_limit=9.4,
            # saturation_effort=940.4,
            velocity_limit=3033.0, # [deg/s]
            stiffness=100.0,
            damping=1.0,
            friction=0.9,
        ),
        "right_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint"],
            effort_limit=9.4,
            # saturation_effort=940.4,
            velocity_limit=3033.0,
            stiffness=100.0,
            damping=1.0,
            friction=0.9,
        ),
    }
)

class CameraBasedRLSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.01),
        ),
    )

    # AI-Mobility-Park config
    mobility_park = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="usd",
        usd_path=os.environ['HOME'] + "/camera_based_rl_isaac/assets/worlds/usd/mobility_park.usd",
        collision_group=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6
        ),
        debug_vis=False
    )

    # robot 
    mobility: ArticulationCfg = MOBILITY_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Mobility")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Mobility/base_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 0.55),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="opengl",
        ),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=400,
            horizontal_aperture=20.0,
            clipping_range=(0.02, 300)
        ),
        width=480,
        height=300,
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    joint_velocity = mdp.JointVelocityActionCfg(asset_name="mobility", joint_names=["left_wheel_joint", "right_wheel_joint"], scale=1.0)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})

        def __post_init__(self) -> None:
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_mobility_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("mobility", joint_names=["caster_yaw_joint", "left_wheel_joint", "right_wheel_joint", "caster_roll_joint"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )

    reset_waypoint_index = EventTerm(
        func=mdp.reset_wp_idx,
        mode="reset",
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (2) path following robot reward
    running_reward = RewTerm(
        func=mdp.target_path_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("mobility"),
            "waypoints": [
                (1.875, -8.873),
                (38.043, -43.308),
                (98.393, -0.736),
                (64.330, 32.624),
                (33.920, 30.395)
            ],
        },
    )
    # (3) time loss reward config
    # time_reward = RewTerm(
    #     func=mdp.time_loss_reward,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("mobility")},
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
        )

    robot_out_of_course = DoneTerm(
        func=mdp.out_of_course_area,
        params={
            "center_line": [
                (-2.187, 11.075),
                (-2.068, 11.075),
                (-2.216, 9.417),
                (-2.168, 7.992),
                (0.202, -20.024),
                (0.865, -23.515),
                (1.906, -26.231),
                (4.406, -30.519),
                (9.936, -36.250),
                (14.200, -38.478),
                (72.056, -56.604),
                (77.550, -58.040),
                (82.246, -59.230),
                (86.111, -57.703),
                (89.453, -56.711),
                (92.395, -55.221),
                (94.814, -53.563),
                (99.075, -49.210),
                (102.459, -43.237),
                (104.063, -35.890),
                (103.915, -33.450),
                (99.309, 17.631),
                (97.576, 22.856),
                (93.949, 28.612),
                (89.790, 32.436),
                (84.576, 35.063),
                (78.687, 36.400),
                (72.507, 36.135),
                (18.716, 31.860),
                (9.171, 28.860),
                (4.230, 24.858),
                (-0.376, 17.834),
                (-1.892, 12.410),
                (-2.187, 11.075)
            ], 
            "course_width": 9.0,
            "asset_cfg": SceneEntityCfg("mobility")},
        )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass

@configclass
class CameraBasedRLCfg(ManagerBasedRLEnvCfg):
    """Configuration for the wheeled quadruped environment."""

    # Scene settings
    scene: CameraBasedRLSceneCfg = CameraBasedRLSceneCfg(num_envs=1, env_spacing=1.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.episode_length_s = 1000
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.render_interval = self.decimation


def vel_controller(vel_msgs: torch.Tensor) -> torch.Tensor:
    wheel_base = 0.6
    wheel_radius = 0.2

    v = vel_msgs[:, 0]
    w = vel_msgs[:, 1]

    left_vel = v - (w * wheel_base / 2)
    right_vel = v + (w * wheel_base / 2)

    left_w_deg = left_vel / wheel_radius # rad/s
    right_w_deg = right_vel / wheel_radius

    left_w = left_w_deg * 180 / 3.14 # deg/s
    right_w = right_w_deg * 180 / 3.14

    actions = torch.stack([left_w, right_w], dim=1).to(vel_msgs.device)

    return actions


def main():
    """Main function: Launch viewer with only robot and world shown."""
    # create environment configuration
    env_cfg = CameraBasedRLCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # create and reset the scene (without stepping physics)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    sample_vel = torch.tensor([[7.0, 0.0]] * args_cli.num_envs).to(args_cli.device)

    while simulation_app.is_running():
        with torch.inference_mode():

            action = vel_controller(sample_vel)
            print(f"action vel : {action}")

            # step the environment
            obs, rew, terminated, truncated, info = env.step(action)

            simulation_app.update()

            state = env.scene.get_state()
            joint_velocity = state["articulation"]["mobility"]["joint_velocity"]

            print(f"Joint velocity (actual): {joint_velocity}")

    # close the environment and simulation
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()