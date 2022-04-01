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
import csv
import gym
import os
import pybullet
import math
import matplotlib.pyplot as plt
from Model.cartpole_model import NN
import numpy as np
import torch
from utility import *

MAX_ITERATIONS = 200 # Number of updates to your gradient
N_EPISODES = 500 # Number of episodes/rollouts
GAMMA = 0.99 # Discounting factor

def loss_f(args, const_return, returns, log_probs):
    log_probs = torch.tensor(log_probs)
    loss = 0
    if args.model == '2':
        loss = -1 * torch.div(
            torch.sum(
                torch.mul(
                    log_probs,
                    returns
                )
            ),
            len(log_probs)
        )
    elif args.model == '3':
        baseline = None
        sigma = None
        loss = -1 * torch.div(
            torch.sum(
                torch.mul(
                    log_probs, 
                    torch.div(
                        torch.sub(
                            returns, 
                            baseline
                        ), 
                        sigma
                    )
                )
            ), 
            len(log_probs)
        )
    else:
        loss = -1 * torch.div(
            torch.sum(
                torch.mul(
                    log_probs, 
                    const_return
                )
            ),
            len(log_probs)
        )
    return loss

def getReturn(rollout, gamma):
    ret = 0.
    power = 0
    for i in range(len(rollout)):
        ret += (gamma**power)*rollout[i][3]
        power += 1
    return ret

def getReturn_t(rollout, snapshot, gamma):
    # Trajectory. A list of tuples [(s0, a0, s1, r1), (s1, a1, s2, r2), ..., (sH-1, aH-1, sH, rH)]
    ret = 0.
    rollout = rollout[snapshot:]
    power = snapshot
    for i in range(len(rollout)):
        # print(f"reward: {rollout[i][3]}")
        # print(f"gamma: {gamma}")
        # print(f"power: {power}")
        ret += (gamma**power)*rollout[i][3]
        # print(f"return: {ret}")
        power += 1
    return ret

def main(args):
    try:
        env.reset()
    except NameError:
        env = gym.make('CartPole-v1')

    # ___Set Seed___

    # ___Build the Models___
    if torch.cuda.is_available():
        if args.verbose: print("cuda available")
        torch.cuda.set_device(args.device)
    else: 
        if args.verbose: print("cuda NOT available")

    policy = NN()

    # ___Testing NN___
    # obs = env.reset()                     # gets you initial state of agent 
    # print(f"obs: {obs}")                  
    # out = policy.forward(obs)             # ask NN to make a prediction (softmax applied)
    # print(f"out: {out}")
    # distribution = torch.distributions.categorical.Categorical(out) # takes a probabilities and create a prob. dist.
    # print(f"distribution: {distribution}")
    # action = distribution.sample()        # get a sample from distribution
    # print(f"action: {action}")
    # exit(0)

    # ___Setup Loss___ Just call loss_f wherever needed.
    # loss = loss_f() # exit(0)

    #___Load Previously Trained Model If Start Epoch > 0___ Skipping
    model_path = f"./models"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # policy.parameters() = policy.fc.parameters() They are the same thing!
    if torch.cuda.is_available():
        policy.cuda()
        optimizer = torch.optim.Adagrad(list(policy.parameters()), lr=args.learning_rate)
    
    optimizer = torch.optim.Adagrad(list(policy.parameters()), lr=args.learning_rate)

    # if args.start_epoch > 0:
    #     load_opt_state(policy, os.path.join(args.model_path, model_name))
    
    print("training...")
    avgRewards = []
    for iter in range(MAX_ITERATIONS):
        totalRewards = []
        totalLoss = 0
        print(f"Iteration {iter} of {MAX_ITERATIONS}")
        for e in range(N_EPISODES):
            if args.verbose: print(f"Episode {e} of {N_EPISODES}, iteration {iter} of {MAX_ITERATIONS}")
            t = 0
            rollout = [] # Trajectory. A list of tuples [(s0, a0, s1, r1), (s1, a1, s2, r2), ..., (sH-1, aH-1, sH, rH)]
            log_probs = [] # the distribution

            # get initial observation (agent's initial state s0)
            prev_state = env.reset() 

            terminate = False
            while not terminate:
                if not args.fast: env.render()
                # if args.verbose: print(obs)  
                # [ 0.04457668 -0.03367179 -0.01488337 -0.03034673  ]   obs -> state t
                # [ cart pos   cart vel    pole angle  pole ang vel ]

                # ask NN to make a prediction (softmax applied)
                out = policy.forward(prev_state)

                # takes discrete probabilities and create a probability distribution.
                distribution = torch.distributions.categorical.Categorical(out)

                # get a sample from distribution
                action = distribution.sample()
                log_probs.append(distribution.log_prob(action))
                # print(distribution.log_prob(action))
                # exit(0)
                if args.verbose: print(f"action taken: {action}")
                next_state, reward, done, info = env.step(int(action)) # obs -> state t+1

                # Store the states, actions, new states, and rewards
                snapshot = (prev_state, action, next_state, reward)
                # print(f"snapshot; {snapshot}")

                rollout.append(snapshot)
                
                prev_state = next_state

                terminate = done
                t += 1

            if args.verbose: print(f"Episode finished after {t} timesteps.")

            const_return = 0
            returns = []
            if args.model == '1':
                # ___Calculate Constant Return___
                const_return = getReturn(rollout, GAMMA)

            else:
                # ___Calculate Return w.r.t. Time___
                for snapshot in range(len(rollout)):
                    ret = getReturn_t(rollout, snapshot, GAMMA)
                    returns.append(ret)
            
            # Bam!

            # ___Calculate Total Rewards per Episode___
            totalReward = 0
            for i in range(len(rollout)):
                totalReward += rollout[i][3]
            totalRewards.append(totalReward)

            # Double Bam!

            # ___Calculate Loss___
            optimizer.zero_grad()

            const_return = to_var(torch.tensor(const_return))
            returns = to_var(torch.FloatTensor(returns))
            log_probs = to_var(torch.FloatTensor(log_probs))
            
            loss = loss_f(args, const_return, returns, log_probs)
            totalLoss += loss

            loss.backward()
            optimizer.step()

            # Triple Bam!
        
        avgLoss = totalLoss / N_EPISODES

        sumTotalRewards = 0
        for tr in totalRewards:
            sumTotalRewards += tr
        avgReward = sumTotalRewards / N_EPISODES
        avgRewards.append(avgReward)

        if iter % 10 == 0:
            model_name = f"cartpole_q_{args.model}_episode_{args.episodes}_epoch_{iter}.pkl"
            save_state(policy, optimizer, os.path.join(model_path, model_name))

        print(f"iter: {iter},\tavgLoss: {avgLoss},\tavgReward: {avgReward}")

    avgRewardFileName = f"average_reward_q_{args.model}.csv"
    with open(avgRewardFileName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(avgRewards)
    if args.verbose: print(f"\n...average rewards wrote to file: {avgRewardFileName}\n")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 4')
    parser.add_argument('-m', '--model', default='1', choices=['1', '2', '3'], \
        help='Enter 1 for question 1, part 1. Enter 2 for question 1 part 2. Enter 3 for question 1, part 3')
    parser.add_argument('-e', '--episodes', default=500, type=int, help='Number of episodes to execute.')
    parser.add_argument('-l', '--learning-rate', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('-f', '--fast', action='store_true', help='Set to disable live animation.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs.')

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