from concurrent.futures import thread
import numpy as np
from typing import Tuple, List, Dict, Any, Union
import time
import matplotlib.pyplot as plt
import cv2
import threading
from robomail.motion import GotoPoseLive

AVAILABLE_SIM_NAMES: List[str] = ['pick_and_place', 'push', 'stack'] 

MAX_POSITION: np.ndarray = np.array([0.35, 0.35, 0.7, 0.08])
MIN_POSITION: np.ndarray = np.array([-0.35, -0.35, 0.0, 0])

### output_obs: dict['success', 'position (4)', 'velocity(3)', 'desired_goal', 'achieved_goal']

class OUNoise:
    def __init__(self, theta, sigma, size, burn_in = 50):
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.x = np.zeros(size)

        for _ in range(burn_in):
            self.step()

    def step(self):
        dx = self.theta*(0-self.x) + self.sigma*np.random.randn(self.size)
        self.x += dx
        return self.x
    
class SuccessEstimator:
    def __init__(self, task_name:str) -> None:
        self.is_grasping: bool = False
        self.bottom_block_in_place: bool = False # for stack task
        self.task_name: str = task_name
        self.run_listener: bool = True
        threading.Thread(target=self.input_listener, daemon=True).start()

    def input_listener(self) -> None:
        print('starting input listener')
        while self.run_listener:
            if self.is_grasping:
                if self.task_name == "stack":
                    in_txt: str = input('Enter p if the block as been placed and released. If the block is realeased outside fo the goal, enter r.')
                    if in_txt == 'p' or 'P':
                        self.bottom_block_in_place = True
                        self.is_grasping = False
                    
                    elif in_txt == 'r' or 'R':
                        self.bottom_block_in_place = False
                        self.is_grasping = False

                    else:
                        print('Invalid input. Please enter p or r.')
                        
            else:
                input('Press enter when the robot has grasped the block')
                self.is_grasping = True
            
            print('is_grasping:', self.is_grasping)
            if self.task_name == "stack":
                print('bottom_block_in_place:', self.bottom_block_in_place)
    
    def __call__(self, goal_location, gripper_location) -> bool:
        return self.is_grasping and np.linalg.norm(goal_location - gripper_location) < 0.04 and (not self.task_name == "stack" or self.bottom_block_in_place)

    def stop(self):
        self.run_listener = False
            

class CustomEnv:
    def __init__(self, task_name:str, inject_noise:bool, random_start:bool = False, K_POS:float = 15.0, K_GRIP:float = 3.0) -> None:
        self.task_name: str = task_name
        self.inject_noise: bool = inject_noise
        self.random_start: bool = random_start

        self.K_POS = K_POS
        self.K_GRIP = K_GRIP

        self.DT = 0.04 # 25 Hz simulation

        self.render_goal = False

        self.FrankaController = GotoPoseLive()

        # check to makes sure the specified task is a valid sim task:
        assert task_name in AVAILABLE_SIM_NAMES, f"TASK_NAME: {task_name} is not a valid sim task. Please choose from: {AVAILABLE_SIM_NAMES}"

        self.reward: int = -1 # default reward
        self.MAX_REWARD:int = 0 # default max reward
        if task_name in ['pick_and_place', 'push']:
            self.MAX_TIME_STEPS: int = 50

        elif task_name == 'stack':
            self.MAX_TIME_STEPS: int = 100

        else:
            raise NotImplementedError(f"Task: {task_name} is not implemented yet.")
        
        # Start the success estimator
        self.success_estimator = SuccessEstimator(task_name)

        # Enviroment parameters
        self.block_size = 0.04
        self.block_xy_range = 0.3
        self.block_z_range = 0.02
        self.center_position = np.array([0.6, 0, self.block_size/2])

        self.reset()

    def sample_state(self) -> Tuple(np.ndarray, np.ndarray):
        """
        Samples a random state for the enviroment. Returns the goal and block position.
        :return: goal, block_position
        """
        # loop until the state is valid
        while True:
            # Sample Goal
            if self.task_name == "push":
                goal = self.center_position + np.random.uniform(low=[-self.block_xy_range, -self.block_xy_range, 0],
                                                                high=[self.block_xy_range, self.block_xy_range, 0])

            elif self.task_name == "pick and place":
                goal = self.center_position + np.random.uniform(low=[-self.block_xy_range, -self.block_xy_range, 0],
                                                                high=[self.block_xy_range, self.block_xy_range, self.block_z_range])

                # 30% of the time the goal is on the ground in the pick and place task.
                if np.random.uniform(0, 1) < 0.3:
                    goal[2] = self.center_position[2]
            
            elif self.task_name == "stack":
                goal = np.empty(6)
                goal[:3] = self.center_position + np.random.uniform(low=[-self.block_xy_range, -self.block_xy_range, 0],
                                                                    high=[self.block_xy_range, self.block_xy_range, 0])  
                goal[3:] = self.goal[0] + np.array([0, 0, self.block_size])

            # Sample inital block location:
            if self.task_name in ["push", "pick_and_place"]:
                block_position = self.center_position + np.random.uniform(low=[-self.block_xy_range, -self.block_xy_range, 0],
                                                                            high=[self.block_xy_range, self.block_xy_range, 0])
                
            elif self.task_name == "stack":
                block_position = np.empty(6)
                block_position[:3] = self.center_position + np.random.uniform(low=[-self.block_xy_range, -self.block_xy_range, 0],
                                                                                high=[self.block_xy_range, self.block_xy_range, 0])
                
                block_position[:3] = self.center_position + np.random.uniform(low=[-self.block_xy_range, -self.block_xy_range, 0],
                                                                            high=[self.block_xy_range, self.block_xy_range, 0])
            
            # Check to make sure the block and goal are not too close together.
            if self.task_name in ["push", "pick and place"]:
                if np.linalg.norm(goal[:3] - block_position) > 0.05:
                    break
            
            elif self.task_name == "stack":
                if np.linalg.norm(goal[:3] - block_position[:3]) > 0.05 and np.linalg.norm(goal[3:] - block_position[3:]) > 0.05 and np.linalg.norm(block_position[:3] - block_position[3:]) > 0.05:
                    break

        return goal, block_position

    
    def inialize_blocks(self, block_position: np.ndarray) -> None:
        # will add some reset logic using franka, potentially. 
        if self.task_name in ["push", "pick and place"]:
            input('move block to ' + str(block_position) + " then press enter.")
        
        elif self.task_name == "stack":
            input('move block 1 to ' + str(block_position[:3]) + '\nmove block 2 to ' + str(block_position[3:]) + " then press enter.")



    def get_obs(self) -> Dict[str, np.ndarray]:
        success = self.success_estimator(self.env_dict["desired_goal"][:3], self.env_dict["observation"][:3])
        if success:
            if input('success detected! If this is not a success, enter n. Otherwise, press enter.') == 'n':
                success = False
            
        position: np.ndarray = np.zeros(3) # TODO: get position from franka
        velocity: np.ndarray = np.zeros(3)
        camera_images = {}
        # Insert realsense code.

        obs: Dict[str, np.ndarray] = {"success": success, 
                                "position": position, 
                                "velocity": velocity,
                                "images": camera_images,
                                "reward": int(success) - 1}
    

        # if self.render_goal:
        #     if self.render:
        #         for camera in self.camera_names:
        #             cv2.imshow(camera, display_images[camera])
        #         if cv2.waitKey(1) & 0xFF == ord('q'): 
        #             exit() # kill on q press
        # else:
        #     if self.render:
        #         for camera in self.camera_names:
        #             cv2.imshow(camera, rendered_images[camera])
        #         if cv2.waitKey(1) & 0xFF == ord('q'): 
        #             exit() # kill on q press
        
        return obs
    
    def reset(self) -> Dict[str, np.ndarray]:
        self.goal, block_position = self.sample_state()
        self.inialize_blocks(block_position)        
    
    def step_pos(self, goal_pose: np.ndarray) -> Dict[str, np.ndarray]:
        # Take a step using the position controller
        # TODO: send the goal pose to the position controller.
        time.sleep(self.DT)
        return self.get_obs()
        
    def normalize_pos(self, pose: np.ndarray) -> np.ndarray:
        return (pose - MIN_POSITION)/(MAX_POSITION - MIN_POSITION)
    
    def unnormalize_pos(self, pose: np.ndarray) -> np.ndarray:
        return pose*(MAX_POSITION - MIN_POSITION) + MIN_POSITION
    
    def step_normalized_pos(self, goal_pose: np.ndarray) -> Dict[str, np.ndarray]:
        # Take a step using the normalized position controller
        goal_pose = self.unnormalize_pos(goal_pose)
        return self.step_pos(goal_pose)
    
    def normalize_grip(self, pose: np.ndarray) -> np.ndarray:
        out_pose = np.copy(pose)
        out_pose[3] = (out_pose[3] - MIN_POSITION[3])/(MAX_POSITION[3] - MIN_POSITION[3])
        return out_pose
    
    def unnormalize_grip(self, pose: np.ndarray) -> np.ndarray:
        out_pose = np.copy(pose)
        out_pose[3] = out_pose[3]*(MAX_POSITION[3] - MIN_POSITION[3]) + MIN_POSITION[3]
        return out_pose
    
    def step_normalized_grip(self, goal_pose: np.ndarray) -> Dict[str, np.ndarray]:
        # Take a step using the normalized position controller
        goal_pose = self.unnormalize_grip(goal_pose)
        return self.step_pos(goal_pose)