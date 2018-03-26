import numpy as np

'''
a simple frame-skipping technique
More precisely, the agent sees and selects actions on every kth frame
instead of every frame, and its last action is repeated on skipped frames.
k = 4

'''


class video_env(object):
    def __init__(self, env):
        self.env = env
        self.k = 4

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        screenRGB, r, done, info = self.env.step(action)
        reward = r
        for i in range(self.k - 1):
            prevScreenRGB = screenRGB
            if done:
                break
            screenRGB, r, done, info = self.env.step(action)
            reward += r

        maxedScreen = np.maximum(screenRGB, prevScreenRGB)

        return maxedScreen, reward, done, info
