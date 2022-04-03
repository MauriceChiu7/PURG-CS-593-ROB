"""
env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
If `rand_init=True`, then the arm will initialize to a random location after each reset.
"""

import argparse
import csv
import gym
import os
import pybullet as p
import time
import math
from Model.reacher_model import NN
import numpy as np
import random
import torch
from utility import *

MAX_ITERATIONS = 1 # Number of updates to your gradient
N_EPISODES = 1 # Number of episodes/rollouts
GAMMA = 0.9 # Discounting factor

def loss_f1(const_return, log_probs):
    # log_probs = torch.FloatTensor(log_probs) # WILL LOSE grad_fn
    # log_probs = torch.tensor(log_probs) # WILL LOSE grad_fn
    # if args.verbose: print(f"===> log_probs[0]:\t", log_probs[0])

    # print("\nCalculating loss with function 1\n")
    loss = 1 * torch.sum(
            torch.mul(
                log_probs, 
                const_return
            )
        )
    # print(loss)
    return loss

def loss_f2(args, returns, log_probs):
    # log_probs = torch.FloatTensor(log_probs)
    loss = 0
    if args.model == '2':
        # print("\nCalculating loss with function 2\n")
        loss = 1 * torch.sum(
                torch.mul(
                    log_probs,
                    returns
                )
            )
    elif args.model == '3':
        # print("\nCalculating loss with function 3\n")
        baseline = torch.mean(returns)
        if args.verbose: print(f"===> baseline:\t\t", baseline)
        sigma = torch.std(returns)
        if args.verbose: print(f"===> sigma:\t\t", sigma)
        loss = 1 * torch.sum(
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
            )
    else:
        loss = None
        print("Wrong input. Terminates")
        exit(0)
    # print(loss)
    return loss

def getReturn(rollout, gamma):
    ret = torch.tensor(0.)
    power = 0
    for i in range(len(rollout)):
        ret += (gamma**power)*rollout[i][3]
        power += 1
    if args.verbose: print(f"===> ret:\t\t", ret)
    return ret

def getReturn_t(rollout, snapshot, gamma):
    # Trajectory. A list of tuples [(s0, a0, s1, r1), (s1, a1, s2, r2), ..., (sH-1, aH-1, sH, rH)]
    ret = torch.tensor(0.)
    rollout = rollout[snapshot:]
    power = snapshot
    for i in range(len(rollout)):
        ret += (gamma**power)*rollout[i][3]
        power += 1
    if args.verbose: print(f"===> ret:\t\t", ret)
    return ret

def main(args):
    try:
        env.reset()
    except NameError:
        if args.random_start: print("rand_init: ", args.random_start)
        env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=args.random_start)

    # print("observation_space: ", env.observation_space)
    # print("action_space: ", env.action_space)
    # prev_state = env.reset() 
    # print("prev_state:", prev_state)
    
    # [ 0.3928371 0.3928371 -0.68091764       0.26561381        0.5          0.         0.08333333   0.       ]
    # [ target_x, target_y, to_target_vec[0], to_target_vec[1], theta/np.pi, theta_dot, gamma/np.pi, gamma_dot]
            
    # next_state, reward, done, info = env.step(np.array([0.5,0.5]))
    # print("next_state: ", next_state)
    # print("reward: ", reward)
    # print("done: ", done)
    # exit(0)

    # ___Set Seed___

    # ___Build the Models___
    if torch.cuda.is_available():
        print("cuda available")
        torch.cuda.set_device(args.device)
    else: 
        print("cuda NOT available")

    policy = NN()

    model_path = f"./models"
    model_name = f"reacher_lossfunc_{args.model}_episode_{args.episodes}_epoch_{args.iterations}.pkl"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print(f"\npolicy.state_dict BEFORE load net state:\n{policy.state_dict()}\n")

    load_net_state(policy, os.path.join(model_path, model_name))

    print(f"\npolicy.state_dict AFTER load net state:\n{policy.state_dict()}\n")
    
    print("testing...")
    avgRewards = []
    successRates = []
    if args.verbose: print(f"===> avgRewards:\t", avgRewards)
    # for iter in range(args.iterations):
    for iter in range(MAX_ITERATIONS):
        totalRewards = []
        totalLoss = torch.tensor(0.)
        # print(f"Iteration {iter} of {args.iterations}")
        success = torch.tensor(0.)
        for e in range(N_EPISODES):
            # set seed
            torch_seed = np.random.randint(low=0, high=1000)
            np_seed = np.random.randint(low=0, high=1000)
            py_seed = np.random.randint(low=0, high=1000)
            torch.manual_seed(torch_seed)
            np.random.seed(np_seed)
            random.seed(py_seed)

            if args.verbose: print(f"Episode {e} of {N_EPISODES}, iteration {iter} of {args.iterations}")
            t = 0
            rollout = [] # Trajectory. A list of tuples [(s0, a0, s1, r1), (s1, a1, s2, r2), ..., (sH-1, aH-1, sH, rH)]
            log_probs = [] # the distribution

            if not args.fast: env.render()
            
            # get initial observation (agent's initial state s0)
            # print("--v env.reset() v--")
            prev_state = env.reset() 
            # print("--^ env.reset() ^--")

            # print(f"\npolicy.state_dict at iter {iter}, episode {e}:\n{policy.state_dict()}\n")

            p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=(0,0,0))
            terminate = False
            
            while not terminate:
                
                # if args.verbose: print(obs)  

                # ask NN to make output 2 means
                mu = policy.forward(prev_state)
                if args.verbose: print(f"===> mu:\t\t", mu)
                # print(f"===> mu:\t\t", mu)

                cov = torch.mul(torch.eye(len(mu)), 1e-2)
                if args.verbose: print(f"===> cov:\n", cov)
                # print(f"===> cov:\n", cov)

                # Use the 2 means and a constant cov matrix to construct two normal dists.
                distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
                # if args.verbose: print(f"===> distribution\t", distribution)

                # get a sample from each normal distribution and constuct them as actions to pass to env.step()
                action = distribution.sample()
                if args.verbose: print(f"===> action:\t", action)
                # print(f"===> action:\t", action)

                # action_prob = out[(action)]
                # if args.verbose: print(f"===> action_prob:\t", action_prob)

                # log_of_ap = torch.log(action_prob)
                # if args.verbose: print(f"===> log_of_ap:\t\t", log_of_ap)
                
                logp = distribution.log_prob(action)
                if args.verbose: print(f"===> logp:\t\t", logp)
                # print(f"===> logp:\t\t", logp)
                # print("================================")

                log_probs.append(logp)
                # log_probs.append(log_of_ap)
                # if args.verbose: print(f"===> log_probs: \t", log_probs) # correct!

                next_state, reward, done, info = env.step(action) # obs -> state t+1
                
                # exit(0)
                if args.slow: time.sleep(1./50.)
                
                # Store the states, actions, new states, and rewards
                snapshot = (prev_state, action, next_state, reward)

                rollout.append(snapshot)
                
                prev_state = next_state

                terminate = done

                t += 1

            if t < 150:
                success += 1
            

            if args.verbose: print(f"Episode finished after {t} timesteps.")
            print(f"Episode finished after {t} timesteps.")

            const_return = torch.tensor(0.)
            if args.verbose: print(f"===> const_return:\t", const_return)
            returns = []
            if args.model == '1':
                # ___Calculate Constant Return___
                const_return = getReturn(rollout, GAMMA)
                if args.verbose: print(f"===> const_return:\t", const_return)
            else:
                # ___Calculate Return w.r.t. Time___
                for snapshot in range(len(rollout)):
                    ret = getReturn_t(rollout, snapshot, GAMMA)
                    if args.verbose: print(f"===> ret:\t\t", ret)
                    returns.append(ret)
                if args.verbose: print(f"===> returns:\n", returns)
            
            # Bam!

            # ___Calculate Total Rewards per Episode___
            totalReward = torch.tensor(0.)
            for i in range(len(rollout)):
                totalReward += rollout[i][3]
            
            totalRewards.append(totalReward)
            if args.verbose: print(f"===> totalRewards:\t", totalRewards)

            # Double Bam!
            
            loss = torch.tensor(0.)
            log_probs = torch.stack(log_probs)
            
            if len(returns) > 0:
                returns = torch.stack(returns)

            if args.model == '1':
                # const_return = to_var(torch.FloatTensor(const_return))
                if args.verbose: print(f"===> const_return:\t", const_return)
                if args.verbose: print(f"===> log_probs:\t", log_probs)
                loss = loss_f1(const_return, log_probs)
                if args.verbose: print(f"===> loss:\t\t", loss)
            else:
                # returns = to_var(torch.FloatTensor(returns))
                if args.verbose: print(f"===> returns:\n", returns)
                loss = loss_f2(args, returns, log_probs)
                if args.verbose: print(f"===> loss:\t\t", loss)
                
            totalLoss += loss
            
            # Triple Bam!
        if args.verbose: print(f"===> totalLoss:\t\t", totalLoss)
        
        successRate = success / N_EPISODES
        successRates.append(successRate)
        print(f"Iter: {iter}: num of successful episodes: {success} over number of episodes: {N_EPISODES}. Success rate = {successRate}")

        avgLoss = totalLoss / N_EPISODES
        # avgLoss = torch.FloatTensor(avgLoss)
        if args.verbose: print(f"===> avgLoss:\t\t", avgLoss)
        # print("\n\n==================\n\n==================\n\n==================\n\n")

        sumTotalRewards = 0
        for tr in totalRewards:
            sumTotalRewards += tr
        avgReward = sumTotalRewards / N_EPISODES
        avgRewards.append(avgReward)

        # if (iter+1) % 10 == 0:
        #     print(f"model at {iter+1} saved")
        #     model_name = f"reacher_lossfunc_{args.model}_episode_{args.episodes}_epoch_{iter+1}.pkl"
        #     save_state(policy, optimizer, os.path.join(model_path, model_name))
        
        # optimizer.zero_grad()
        # avgLoss.backward()
        # optimizer.step()

        # print(f"\npolicy.state_dict:\n{policy.state_dict()}\n")
        # print("optimizer.step() called!")

        print(f"Iter: {iter},\tavgLoss: {avgLoss},\tavgReward: {avgReward}")

    successRateFileName = f"reacherTestSuccess_rate_model_{args.model}_episodes_{args.episodes}_epochs_{args.iterations}.csv"
    with open(successRateFileName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(successRates)
    print(f"\n...success rates wrote to file: {successRateFileName}\n")

    avgRewardFileName = f"reacherTestAverage_reward_model_{args.model}_episodes_{args.episodes}_epochs_{args.iterations}.csv"
    with open(avgRewardFileName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(avgRewards)
    print(f"\n...average rewards wrote to file: {avgRewardFileName}\n")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 4')
    parser.add_argument('-m', '--model', default='1', choices=['1', '2', '3'], \
        help='Enter 1 for question 1, part 1. Enter 2 for question 1 part 2. Enter 3 for question 1, part 3')
    parser.add_argument('-i', '--iterations', default=200, type=int, help='Number of iterations/epochs.')
    parser.add_argument('-e', '--episodes', default=500, type=int, help='Number of episodes to execute.')
    parser.add_argument('-r', '--random-start', action='store_true', help='Print logs.')
    parser.add_argument('-f', '--fast', action='store_true', help='Set to disable live animation.')
    parser.add_argument('-s', '--slow', action='store_true', help='Play live animation in slow motion.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs.')

    args = parser.parse_args()

    main(args)