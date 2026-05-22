import numpy as np
import time
import logging
import gymnasium as gym


log = logging.getLogger(__name__)


try:
    from pynput import keyboard
except ModuleNotFoundError:
    log.warning("pynput module not found. Please install it to use keyboard controls.")
    keyboard = None


class InteractiveMaze:
    def __init__(self, **kwargs):
        self.env = gym.make('Maze-v0', render_mode='human', **kwargs)
        self.action = None  # No action until key pressed
        self.running = True
        self.listener = None
        self.step_mode = True  # Wait for input between steps
        
    def on_press(self, key):
        """Handle keyboard press events"""
        if isinstance(key, keyboard.Key):
            if key == keyboard.Key.up:
                self.action = 0  # Move up
                log.debug("Action: UP (0)")
            elif key == keyboard.Key.right:
                self.action = 1  # Move right
                log.debug("Action: RIGHT (1)")
            elif key == keyboard.Key.down:
                self.action = 2  # Move down
                log.debug("Action: DOWN (2)")
            elif key == keyboard.Key.left:
                self.action = 3  # Move left
                log.debug("Action: LEFT (3)")
            elif key == keyboard.Key.esc:
                print("\nExiting...")
                self.running = False
                return False  # Stop listener
        elif hasattr(key, 'char'):
            if key.char == 'r':
                log.info("Resetting environment")
                self.action = 'reset'
        else:
            log.warning(f"Unrecognized key type: {key=}, {type(key)=}")
    
    def log_state(self, observation, reward, step, total_reward):
        """Print current state information"""
        agent_pos_flat = observation[1].argmax()
        agent_row, agent_col = np.unravel_index(agent_pos_flat, observation[1].shape)
        
        goal_pos_flat = observation[2].argmax()
        goal_row, goal_col = np.unravel_index(goal_pos_flat, observation[2].shape)
        
        log.debug("Maze State:"
                  f"\nStep {step}: Agent at ({agent_row}, {agent_col})"
                  f"\nGoal at ({goal_row}, {goal_col})"
                  f"\nStep reward: {reward:.3f}"
                  f"\nTotal reward: {total_reward:.3f}")
        
        # Calculate Manhattan distance to goal
        distance = abs(agent_row - goal_row) + abs(agent_col - goal_col)
        log.debug(f"Distance to goal: {distance}")

    def run(self):
        """Main game loop"""
        print("=" * 50)
        print("Interactive Maze Environment")
        print("=" * 50)
        print("\nControls:")
        print("  UP ARROW    - Move up")
        print("  DOWN ARROW  - Move down")
        print("  LEFT ARROW  - Move left")
        print("  RIGHT ARROW - Move right")
        print("  R           - Reset environment")
        print("  ESC         - Quit")
        print("\nWalls are black, free space is white.")
        print("You are the blue circle.")
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
        observation, _ = self.env.reset()
        self.log_state(observation, 0.0, 0, 0.0)
        
        total_reward = 0
        step = 0
        
        try:
            while self.running:
                # Wait for action input
                if self.action is None:
                    time.sleep(0.05)
                    continue
                
                # Handle reset action
                if self.action == 'reset':
                    observation, _ = self.env.reset()
                    total_reward = 0
                    step = 0
                    self.action = None
                    self.log_state(observation, 0.0, 0, 0.0)
                    continue
                
                # Take action in environment
                observation, reward, terminated, truncated, _ = self.env.step(self.action)
                total_reward += reward
                step += 1
                
                self.log_state(observation, reward, step, total_reward)
                
                # Reset action to wait for next input
                self.action = None
                
                if terminated or truncated:
                    print("\n" + "=" * 50)
                    if terminated:
                        print("🎉 Goal reached! Congratulations!")
                    print(f"Episode finished!")
                    print(f"Total steps: {step}")
                    print(f"Total reward: {total_reward:.2f}")
                    print("=" * 50)
                    print("\nAutomatically resetting environment...")
                    time.sleep(0.2)
                    
                    # Automatic reset
                    observation, _ = self.env.reset()
                    total_reward = 0
                    step = 0
                    self.action = None
                    self.log_state(observation, 0.0, 0, 0.0)
                
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            if self.listener:
                self.listener.stop()
            self.env.close()
            log.info("Environment closed")

if __name__ == '__main__':
    game = InteractiveMaze(size=20)
    game.run()

