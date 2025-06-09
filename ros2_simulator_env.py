import os
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # ✅ この時点で Isaac Sim が立ち上がる

import torch

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.sim import DomeLightCfg, GroundPlaneCfg
from isaaclab.assets import AssetBaseCfg
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelController(Node):
    def __init__(self, robot, wheel_separation=0.7, wheel_radius=0.247):
        super().__init__('cmd_vel_controller')
        self.robot = robot
        self.wheel_separation = wheel_separation
        self.wheel_radius = wheel_radius

        self.num_envs = 1

        self.joint_ids, _ = self.robot.find_joints(["left_wheel_joint", "right_wheel_joint"])

        self.device = self.robot.device  # IsaacLab の device (cuda/cpu)
        self.env_ids = torch.tensor([0], device=self.robot.device)

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.subscription

    def cmd_vel_callback(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z

        v_l = (v - w * self.wheel_separation / 2.0) / self.wheel_radius
        v_r = (v + w * self.wheel_separation / 2.0) / self.wheel_radius

        joint_ids = self.joint_ids

        target_velocities = torch.zeros((self.num_envs, len(joint_ids)), device=self.device)
        target_velocities[:, 0] = v_l*1e4
        target_velocities[:, 1] = v_r*1e4

        print(target_velocities)
        self.robot.set_joint_velocity_target(target_velocities, joint_ids=joint_ids)

@configclass
class TeleopSimSceneCfg(InteractiveSceneCfg):
    """構築された手動操縦環境の設定クラス。"""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(intensity=3000.0)
    )

    my_robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=os.path.expanduser("~/Documents/robot_model/param_fix_mobility.usd"),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=True),
            articulation_props=ArticulationRootPropertiesCfg(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),
            rot=(0, 0, 0, 1),
            joint_pos={"caster_yaw_joint": 0.0},
        ),
        actuators={
            "left_wheel_actuator": DCMotorCfg(
                joint_names_expr=["left_wheel_joint"],
                effort_limit=770.4,
                saturation_effort=1640.4,
                velocity_limit=4749.82, # [deg/s]
                stiffness=0.0,
                damping=1.0,
                friction=0.9,
            ),
            "right_wheel_actuator": DCMotorCfg(
                joint_names_expr=["right_wheel_joint"],
                effort_limit=770.4,
                saturation_effort=1640.4,
                velocity_limit=4749.82,
                stiffness=0.0,
                damping=1.0,
                friction=0.9,
            ),
            "caster_yaw_actuator": IdealPDActuatorCfg(
                joint_names_expr=["caster_yaw_joint"],
                effort_limit=100.5,
                velocity_limit=None,
                stiffness=5.0,
                damping=10.0,
                friction=0.2,
            ),
        }
    )

    mobility_park = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="usd",
        usd_path=os.environ['HOME'] + "/camera_based_rl_isaac/assets/worlds/usd/mobility_park.usd",
        collision_group=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
        ),
        debug_vis=False
    )

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3, 3, 2], target=[0, 0, 0])

    scene_cfg = TeleopSimSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Simulator is ready.")

    # ROSノード起動
    rclpy.init()
    controller = CmdVelController(robot=scene["my_robot"])

    try:
        while simulation_app.is_running():
            sim.step()
            rclpy.spin_once(controller, timeout_sec=0.001)
            scene.update(sim.get_physics_dt())
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()