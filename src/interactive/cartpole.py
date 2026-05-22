import time
import logging
import numpy as np
import gymnasium as gym


log = logging.getLogger(__name__)


try:
    from pynput import keyboard
except ModuleNotFoundError:
    log.warning("pynput module not found. Please install it to use keyboard controls.")
    keyboard = None


class InteractiveCartPole:
    def __init__(self, env_name='CartPole-v1', **kwargs):
        self.env = gym.make(env_name, render_mode='human', **kwargs)
        self.action = 0  # Default action (move left)
        self.running = True
        self.listener = None
        
    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.Key.left:
                self.action = 0  # Push cart to the left
                print("Action: LEFT (0)")
            elif key == keyboard.Key.right:
                self.action = 1  # Push cart to the right
                print("Action: RIGHT (1)")
            elif key == keyboard.Key.esc:
                print("\nExiting...")
                self.running = False
                return False  # Stop listener
        except AttributeError:
            pass
    
    def print_state(self, observation, reward, step):
        """Print current state information"""
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        print(f"\n--- Step {step} ---")
        print(f"Cart Position: {cart_pos:7.3f}")
        print(f"Cart Velocity: {cart_vel:7.3f}")
        print(f"Pole Angle:    {pole_angle:7.3f} rad ({np.degrees(pole_angle):6.2f}°)")
        print(f"Pole Velocity: {pole_vel:7.3f}")
        print(f"Reward:        {reward:7.3f}")
    
    def run(self):
        """Main game loop"""
        print("=" * 50)
        print("Interactive CartPole-v1")
        print("=" * 50)
        print("\nControls:")
        print("  LEFT ARROW  - Push cart left")
        print("  RIGHT ARROW - Push cart right")
        print("  ESC         - Quit")
        print("\nGoal: Balance the pole upright for as long as possible!")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
        observation, info = self.env.reset()
        self.print_state(observation, 0.0, 0)
        
        total_reward = 0
        step = 0
        
        try:
            while self.running:
                # Take action in environment
                observation, reward, terminated, truncated, info = self.env.step(self.action)
                total_reward += reward
                step += 1
                
                self.print_state(observation, reward, step)
                
                if terminated or truncated:
                    print("\n" + "=" * 50)
                    print(f"Episode finished!")
                    print(f"Total steps: {step}")
                    print(f"Total reward: {total_reward:.2f}")
                    print("=" * 50)
                    
                    # Ask to restart
                    print("\nPress 'r' to restart or ESC to quit...")
                    restart = False
                    
                    def on_press_end(key):
                        nonlocal restart
                        if key == keyboard.Key.esc:
                            self.running = False
                            return False
                        try:
                            if key.char == 'r':
                                restart = True
                                return False
                        except AttributeError:
                            pass
                    
                    with keyboard.Listener(on_press=on_press_end) as listener:
                        listener.join()
                    
                    if restart:
                        observation, info = self.env.reset()
                        total_reward = 0
                        step = 0
                        self.action = 0
                        self.print_state(observation, 0.0, 0)
                        self.listener = keyboard.Listener(on_press=self.on_press)
                        self.listener.start()
                    else:
                        break
                
                # Small delay to make it playable
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            if self.listener:
                self.listener.stop()
            self.env.close()
            print("\nEnvironment closed. Goodbye!")


if __name__ == '__main__':
    game = InteractiveCartPole('CartPole-v1')
    game.run()

