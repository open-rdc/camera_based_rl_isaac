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

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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
        usd_path=os.environ['HOME'] + "/camera_based_rl_isaac/assets/robots/mobility/usd/mobility.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=7.0,
            max_angular_velocity=4.0,
            max_depenetration_velocity=5.0,
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
        joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0, "caster_wheel_joint": 0.0},
    ),
    actuators={
        "left_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=12.0,
            damping=10.0,
        ),
        "right_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=12.0,
            damping=10.0,
        ),
    },
)


class CameraBasedRLSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # AI-Mobility-Park config
    mobility_park = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="usd",
        usd_path=os.environ['HOME'] + "/camera_based_rl_isaac/assets/worlds/usd/mobility_park.usd",
        collision_group=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.0,
            dynamic_friction=1.0
        ),
        debug_vis=True
    )

    # robot 
    mobility: ArticulationCfg = MOBILITY_CONFIG.replace(prim_path="/World/envs/env_0/Mobility")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_0/Mobility/chassis/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 0.55),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="opengl",
        ),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=156.08,
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
    joint_velocity = mdp.JointVelocityActionCfg(asset_name="mobility", joint_names=["left_wheel_joint", "right_wheel_joint"], scale=100.0)

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
            "asset_cfg": SceneEntityCfg("mobility", joint_names=["left_wheel_joint", "right_wheel_joint"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (2) Primary task: keep robot running on the line
    running_reward = RewTerm(
        func=mdp.compute_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("tiled_camera")},
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # robot_out_of_course = DoneTerm(
    #     func=mdp.termination,
    #     params={"bounds": (-10.0, 10.0), "asset_cfg": SceneEntityCfg("tiled_camera")},
    # )

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
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.render_interval = self.decimation

def main():
    """Main function: Launch viewer with only robot and world shown."""
    # create environment configuration
    env_cfg = CameraBasedRLCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # create and reset the scene (without stepping physics)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 100 == 0:
                count=0
                env.reset()

                print("-" * 80)
                print("[INFO]: Environment initialized. Displaying scene...")

            joint_vel = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_vel)

            simulation_app.update()

            state = env.scene.get_state()

            root_pose = state["articulation"]["mobility"]["root_pose"]
            print(f"Root pose: {root_pose}")

            count += 1
    # close the environment and simulation
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()