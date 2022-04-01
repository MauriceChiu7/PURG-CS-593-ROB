"""
env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
If `rand_init=True`, then the arm will initialize to a random location after each reset.
"""

# Useful code from Assignment 2
# Resetting q0 to pi and q1 to 0
# env.unwrapped.robot.central_joint.reset_position(np.pi, 0) # reset_position(position, velocity)
# env.unwrapped.robot.elbow_joint.reset_position(0, 0)
# q0, q0_dot = env.unwrapped.robot.central_joint.current_position()
# q1, q1_dot = env.unwrapped.robot.elbow_joint.current_position()
# calculate torque
# tau = np.dot(jT, force)
# apply torque to joints
# env.step(tau)
# mse_x = np.square(np.subtract(x, actual_x)).mean()
# mse_y = np.square(np.subtract(y, actual_y)).mean()
# plt.plot(np.arange(0, len(errors_x)), errors_x, 'b', np.arange(0, len(errors_y)), errors_y, 'r')
# plt.savefig('err_v_iter_x-y.png')
# plt.show()
# plt.clf()

# An episode is a sequence of state/action/reward steps, from the start state until the end. (For the cartpole, the end occurs when the pole falls over, the cart moves off the screen, or after 500 steps). In the context of reinforcement learning, "episode" is synonymous with "trajectory."

# 500 episodes just means that n=500 in the equation given; i.e. your J function should be an average across 500 episodes. The policy remains fixed throughout all these runs.

# 200 iterations means you should perform the gradient update 200 times.

# 1. Implemenet a vanilla reinforce algorithm given by the following gradient update for your policy.

import argparse
import gym
import os
import pybullet
import math
import matplotlib.pyplot as plt
import Model.cartpole_model as model
import numpy as np
import torch
from utility import *

MAX_ITERATION = 200 # Number of updates to your gradient
N_EPISODES = 500 # Number of episodes/rollouts
GAMMA = 0.99 # Discounting factor

def main(args):
    try:
        env.reset()
    except NameError:
        env = gym.make('CartPole-v1')

    # ___Set Seed___

    # ___Build the Models___

    policy = model.NN

    # ___Setup Loss___

    model_path = f"./models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = f"cartpole_{args.episodes}"

    obs = env.reset()
    print(obs)
    out = policy.forward(obs)
    print(out)
    exit(0)

    for iter in range(MAX_ITERATION):
        
        for e in range(N_EPISODES):
            t = 0
            rollout = [] # Trajectory. A list of tuples [(s0, a0, s1, r1), (s1, a1, s2, r2), ..., (sH-1, aH-1, sH, rH)]
            obs = env.reset() # get initial observation (robot's initial state s0)
            terminate = False
            while not terminate:
                if not args.fast: env.render()
                print(obs)  # [ 0.04457668 -0.03367179 -0.01488337 -0.03034673  ]   obs -> state t
                            # [ cart pos   cart vel    pole angle  pole ang vel ]
                cartPos = obs[0]
                cartVel = obs[1]
                poleAng = obs[2]
                poleAnV = obs[3]
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action) # obs -> state t+1
                terminate = done
                # print(f"run: {run}")
                t += 1

            print(f"Episode finished after {t} timesteps\n")
            
                
                # Store the states, actions, new states, and rewards
                
            # Calculate discountedReturn(rollout.reward) w.r.t. time
            # Calculate baseline(R)
            # Calculate loss(reward)

            
    env.close()


    # for gradUpdate in range(200):
    #     for episode in range(500):
    #         pass

    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 4')
    parser.add_argument('-m', '--model', default='1', choices=['1', '2', '3'], \
        help='Enter 1 for question 1, part 1. Enter 2 for question 1 part 2. Enter 3 for question 1, part 3')
    parser.add_argument('-e', '--episodes', default=500, type=int, help='Number of episodes to execute.')
    parser.add_argument('-l', '--learning-rate', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('-f', '--fast', action='store_true', help='Set to disable live animation.')

    args = parser.parse_args()

    main(args)

# load libraries (if this fails, see "Installation Instructions")
# initialize constants

# initialize the environment

# This try-except is to make sure there is only a single pybullet connection set-up
# try:
#     env.reset()
# except NameError:
#     env = gym.make("CartPole-v1")

# env.render(mode="human")
# obs = env.reset()

# while env.done:
#     p.stepSimulation()