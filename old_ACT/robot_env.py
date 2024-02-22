import numpy as np
from typing import Tuple, List, Dict, Any, Union
import time
from robomail.motion import GotoPoseLive
from robomail.vision import NonThreadedCameras
from autolab_core import RigidTransform, transformations

# May need to change this
MAX_POSITION: np.ndarray = np.array([0.35, 0.35, 0.7, 0.08])
MIN_POSITION: np.ndarray = np.array([-0.35, -0.35, 0.2, 0])

class RobotEnv:
    # camera numbers is list of either ints or strings
    def __init__(self, camera_numbers:List[Union[int, str]] = [1, 2, 3, 4, 5],
                 use_gelsight = True,
                 image_height=480,
                 image_width=848,
                 inject_noise:bool = False, 
                 random_start:bool = False, 
                 dt:float = 0.1,
                 min_gripper_width:float = 0.00675,
                 ) -> None:
        
        self.inject_noise: bool = inject_noise
        self.random_start: bool = random_start

        self.dt: float = dt

        self.min_gripper_width: float = min_gripper_width
        
        # Set up cameras
        # if camera numbers are strings, convert them to ints
        camera_numbers = [int(cam_num) for cam_num in camera_numbers]
        self.cameras = NonThreadedCameras(camera_numbers, image_height=image_height, 
                                          image_width=image_width, get_point_cloud=False, 
                                          get_verts=False)
        self.image_height = image_height
        self.image_width = image_width
        self.camera_numbers = camera_numbers
        self.use_gelsight = use_gelsight

        if use_gelsight:
            from simple_gelsight import Gelsight, get_camera_id
            self.GelSight = Gelsight(get_camera_id('GelSight'), use_gpu=True)
        else:
            self.GelSight = None


        self.FrankaController = GotoPoseLive()
        self.reset()
        self.last_time = time.time()

    
    def get_obs(self) -> Dict[str, np.ndarray]:
        robot_data = self.FrankaController.fa.get_robot_state()
        current_pose: RigidTransform  = robot_data['pose']*self.FrankaController.fa._tool_delta_pose
        cur_joints: RigidTransform = robot_data['joints']
        cur_vel: np.ndarray = robot_data['joint_velocities']
        finger_width: float = robot_data['gripper_width']
        jacobian: np.ndarray = self.FrankaController.fa.get_jacobian(cur_joints)

        current_relative_rotation: Tuple[float, float, float] = (current_pose*self.FrankaController.FC.HOME_POSE.inverse()).euler_angles
        current_velocity_arm: np.ndarray = jacobian@cur_vel # get the current velocity of the arm using the jacobian
        

        current_pose_info = np.concatenate((current_pose.translation, current_relative_rotation, np.array([finger_width])))
        velocity_info = np.concatenate((current_velocity_arm, np.array([0])))

        # Get camera images
        frames = self.cameras.get_next_frames()
        camera_images: Dict[int, np.ndarray] = {}
        for i, cam_num in enumerate(self.camera_numbers):
            camera_images[str(cam_num)] = frames[i][0] # key is camera name

        # record to observation dictionary:
        obs: Dict[str, np.ndarray] = {"success": False, # assume that the goal is not reached
                                      "position": current_pose_info, 
                                      "velocity": velocity_info,
                                      "images": camera_images,
                                      "pose": current_pose,
                                      "reward": 0,}
              
        if self.use_gelsight:
            frame, marker_data, depth, strain_x, strain_y = self.GelSight.get_frame()
            obs["gelsight"] = {"raw_image": frame,
                               "marker_data": marker_data,
                               "depth_strain_image": np.stack((depth, strain_x, strain_y), axis=-1)}
        
        return obs
    
    def reset(self) -> Dict[str, np.ndarray]:
        # First, have the user move the robot to a save position
        input('Episode has ended. Please move the robot to a safe position and press enter to continue.')
        # Reset the robot
        # For now, just move the hand back to the home pose.
        self.FrankaController.step(self.FrankaController.FC.HOME_POSE)
        self.FrankaController.fa.open_gripper(block=False)
        input("Reseting the robot to the home position, press enter to continue.")
        return self.get_obs()
        
    
    def step_pose_info(self, goal_pose_info: np.ndarray, current_pose:RigidTransform = None) -> Dict[str, np.ndarray]:
        # Take a step using the position controller
        # TODO: send the goal pose to the position controller.
        goal_pose = self.FrankaController.FC.HOME_POSE.copy()
        if goal_pose_info.size == 4: # only position, [x, y, z, grip]
            goal_pose.translation = goal_pose_info[:3]
        elif goal_pose_info.size == 7: # position and rotation, [x, y, z, roll, pitch, yaw, grip]
            goal_pose.translation = goal_pose_info[:3]
            goal_pose.rotation = transformations.euler_matrix(goal_pose_info[3], goal_pose_info[4], goal_pose_info[5])
        else:
            raise ValueError("pose_info must be of size 4 (x, y, z, grip) or 7 (x, y, z, roll, pitch, yaw, grip)")

        # Step the controller
        self.FrankaController.step(goal_pose, current_pose)

        # Move the gripper, make sure that the gripper width is not too small
        self.FrankaController.fa.goto_gripper(max(goal_pose_info[-1], self.min_gripper_width))

        # Wait (if necessary) to ensure a constant control loop time
        control_loop_time = time.time() - self.last_time
        print(f"Control loop time: {control_loop_time}")
        time.sleep(max(0, self.dt - control_loop_time))
        self.last_time = time.time()

        return self.get_obs()
        
    def normalize_pose_info(self, pose_info: np.ndarray) -> np.ndarray:
        if pose_info.size == 4: # only position, [x, y, z, grip]
            return (pose_info - MIN_POSITION)/(MAX_POSITION - MIN_POSITION)
        if pose_info.size == 7: # position and rotation, [x, y, z, roll, pitch, yaw, grip]
            normalized_pose_info: np.ndarray = np.copy(pose_info)
            normalized_pose_info[:3] = (pose_info[:3] - MIN_POSITION[:3])/(MAX_POSITION[:3] - MIN_POSITION[:3])
            normalized_pose_info[6] = (pose_info[6] - MIN_POSITION[3])/(MAX_POSITION[3] - MIN_POSITION[3])
            return normalized_pose_info
        else:
            raise ValueError("pose_info must be of size 4 (x, y, z, grip) or 7 (x, y, z, roll, pitch, yaw, grip)")
    
    def unnormalize_pose_info(self, norm_pose_info: np.ndarray) -> np.ndarray:
        if norm_pose_info.size == 4:
            return norm_pose_info*(MAX_POSITION - MIN_POSITION) + MIN_POSITION
        if norm_pose_info.size == 7:
            unnormalized_pose_info: np.ndarray = np.copy(norm_pose_info)
            unnormalized_pose_info[:3] = norm_pose_info[:3]*(MAX_POSITION[:3] - MIN_POSITION[:3]) + MIN_POSITION[:3]
            unnormalized_pose_info[6] = norm_pose_info[6]*(MAX_POSITION[3] - MIN_POSITION[3]) + MIN_POSITION[3]
            return unnormalized_pose_info
        else:
            raise ValueError("pose_info must be of size 4 (x, y, z, grip) or 7 (x, y, z, roll, pitch, yaw, grip)")
    
    def step_normalized_pose_info(self, norm_goal_pose: np.ndarray, current_pose: RigidTransform = None) -> Dict[str, np.ndarray]:
        # Take a step using the normalized position controller
        goal_pose_info:np.ndarray = self.unnormalize_pose_info(norm_goal_pose)
        return self.step_pose_info(goal_pose_info, current_pose)