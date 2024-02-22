from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


from panda_gym.envs.core import Task
from panda_gym.utils import distance



class PegInsertion(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.02,
        goal_xy_range: float = 0.3,
        obj_xy_range: float = 0.3,
        hole_size: float = 0.05,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.hole_size = hole_size
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, self.hole_size/2])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, self.hole_size/2])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self._create_hole(np.array([0.0, 0.0]))

        # Create hole by creating 4 recngular prizoms around the hole.
        # self.sim.create_box(
        #     body_name="target",
        #     half_extents=np.ones(3) * self.object_size / 2,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.array([0.0, 0.0, 0.05]),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        # )
        # need 4 boxes, creating a square with a hole in center
    
    def _create_hole(self, center, move=False) -> None:
        center = np.array([center[0], center[1], self.hole_size/2])
        half_sizes = np.array([[self.hole_size/2, self.hole_size, self.hole_size/2],
                          [self.hole_size/2, self.hole_size, self.hole_size/2],
                          [self.hole_size, self.hole_size/2, self.hole_size/2],
                          [self.hole_size, self.hole_size/2, self.hole_size/2]])
        displacement = np.array([[-self.hole_size, -self.hole_size/2, 0],
                                 [self.hole_size, self.hole_size/2, 0],
                                 [-self.hole_size/2, self.hole_size, 0],
                                 [self.hole_size/2, -self.hole_size, 0]])
        if move:
            for i in range(4):
                self.sim.set_base_pose("hole"+str(i), center + displacement[i], np.array([0.0, 0.0, 0.0, 1.0]))
        else:
            for i in range(4):
                self.sim.create_box(
                    body_name="hole"+str(i),
                    half_extents=half_sizes[i],
                    mass=0.0,
                    ghost=False,
                    position=center + displacement[i],
                    rgba_color=np.array([0.5, 0.5, 0.5, 1]),
                )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        while np.linalg.norm(self.goal-object_position) < 3*self.hole_size:
            object_position = self._sample_object()
            self.goal = self._sample_goal()
        self._create_hole(self.goal[:2], move=True)
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
        


class PandaPegInsertionEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".5
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.

    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        # The existing Panda robot has an issue with the gripper. The joints do not have a max position, causing the gripper to osilate. Set a limit
        # for i in range(2):
        #     robot.sim.physics_client.changeDynamics(robot.sim._bodies_idx[robot.body_name],  robot.fingers_indices[i], jointLowerLimit=0.0, jointUpperLimit=0.04)
        task = PegInsertion(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

if __name__ == "__main__":
    import time
    env = PandaPegInsertionEnv(render_mode="human", control_type="joints")
    env.reset()
    for i in range(1000):
        print(i)
        env.step(env.action_space.sample())
        # env.render()
        time.sleep(1)
    env.close()
    print("Done!")
