from mss import mss
import pydirectinput
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
import pyautogui
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN, PPO
import time

pydirectinput.FAILSAFE = False
pydirectinput.PAUSE=0.00
resizedWidth = 100
resizedHeight = 60

class WebGame(Env):
  
    def __init__(self):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(1,resizedHeight,resizedWidth), dtype=np.uint8)
        self.action_space = Discrete(2)
        # Extraction
        self.cap = mss()
        self.game_location = {'top':400, 'left': 81, 'width': 500, 'height': 300}
        self.done_location = {'top':425, 'left':686, 'width': 5, 'height': 5}

    def step(self, action):
        action_map = {
            0:'up',
            1:'no_op'
        }
        reward = 1
        if action != 1:
            cmd = action_map[int(action)]
            pydirectinput.keyDown(cmd)
            time.sleep(.025)
            pydirectinput.keyUp(cmd)
        else:
            time.sleep(.025)

        # Checking whether the game is done
        done = self.get_done()
        # Get the next observation
        new_observation = self.get_observation()
        # Reward
        
        info = {}

        return new_observation, reward, done, info

    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()

    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (resizedWidth, resizedHeight))
        channel = np.reshape(resized, (1,resizedHeight,resizedWidth))
        return channel
    
    def get_done(self):
        img = pyautogui.screenshot(region=(686, 425, 1, 1))

        done = False
        if (img.getpixel((0,0)) == (83, 83, 83)):
            done = True
        
        return done

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/dino/'
LOG_DIR = './logs/dino/'

callback  = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_DIR)

env = WebGame()
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=.00002, buffer_size=180000, learning_starts=1000)
#model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, learning_rate=.00005, n_steps=512, verbose=1)

model.learn(total_timesteps=300000, callback=callback)
