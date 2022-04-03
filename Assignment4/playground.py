import gym
import pybullet as p
import time

try:
    env.reset()
except NameError:
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)

env.render()
env.reset()

p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=(0,0,0))

env.reset()

while 1:
    env.step([-0.01, -0.01])
    # p.stepSimulation()
    time.sleep(1./25.)
    