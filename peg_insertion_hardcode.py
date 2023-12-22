import numpy as np

class Controller:
    def __init__(self, hole_location, block_width, hole_width, K = 10, max_action=[0.1, 0.1, 0.1, 0.1]) -> None:
        self.hole_location = hole_location
        self.grasped = False
        self.last_action = np.zeros(4)
        self.K = K
        self.block_width = block_width
        self.hole_width = hole_width
        self.max_action = np.array(max_action)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self._check_grasped(obs)
        if not self.grasped:
            action = self._reach(obs)
        else:
            action = self._insert(obs)
        
        # Clip the action
        return np.clip(action, -self.max_action, self.max_action)
        
    def _reach(self, obs: np.ndarray) -> np.ndarray:
        ee_position = obs['observation'][:3]
        block_position = obs['achieved_goal']

        # First, move the gripper above the block
        if np.linalg.norm(block_position[:2] - ee_position[:2]) < self.block_width/1.5:
            goal = block_position
        else:
            goal = np.array([block_position[0], block_position[1], 0.2])
        
        # Use proportional control to reach the block
        action = self.K * (goal - ee_position)

        # If the gripper is close to the block, close the gripper. Else, open it.
        if np.linalg.norm(block_position - ee_position) < self.block_width/1.5:
            action = np.concatenate([action, [-0.1]])
        else:
            action = np.concatenate([action, [0.1]])

        self.last_action = action
        return action

    def _check_grasped(self, obs: np.ndarray) -> bool:
        # Check if the gripper is grasping the block
        if self.last_action[3] > 0:
            self.grasped = False
        
        else:
            if abs(obs['observation'][6] - self.block_width) < 0.01 and np.linalg.norm(obs['observation'][:3] - obs['achieved_goal'][:3]) < self.block_width:
                self.grasped = True
            else:
                self.grasped = False
        
        return self.grasped

        
    def _insert(self, obs: np.ndarray) -> np.ndarray:
        # Check if the block is in the hole. If in the hole, release the block.
        if self._check_in_hole(obs):
            action = np.array([0, 0, 0, 1])
            self.last_action = action
            return action
        
        # If not in the hole, use proportional control to insert the block
        block_position = obs['achieved_goal']
        
        # Move the gripper above the hole
        if np.linalg.norm(self.hole_location[:2] - block_position[:2]) < self.hole_width/2:
            goal = self.hole_location
        else:
            goal = np.array([self.hole_location[0], self.hole_location[1], 0.2])
        action = self.K * (goal - block_position)
        action = np.concatenate([action, [-0.1]])
        self.last_action = action
        return action
    
    def _check_in_hole(self, obs: np.ndarray) -> bool:
        # Check if the block is in the hole
        block_position = obs['achieved_goal']
        block_error = block_position - self.hole_location
        # Make sure the block is aligned with the hole, and the block is in the hole (vertically)
        if np.all(np.abs(block_error[:2]) < self.hole_width) and block_position[2] < self.hole_location[2] + 0.5*(self.block_width + self.hole_width) - 0.25*self.block_width:
            return True
        else:
            return False
        
    
if __name__ == "__main__":
    from peg_insertion_env import PandaPegInsertionEnv
    import time
    env = PandaPegInsertionEnv(render_mode="human", reward_type="sparse", control_type="ee")
    obs, info = env.reset()
    controller = Controller(hole_location=obs['desired_goal'], block_width=0.04, hole_width=0.05)
    for i in range(1000):
        action = controller(obs)
        # print(action, controller.grasped, controller._check_in_hole(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward, terminated, truncated)
        time.sleep(0.05)
    env.close()
    print("Done!")
        
        

        

