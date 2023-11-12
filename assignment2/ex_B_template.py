# -*- coding: utf-8 -*-
# Revised version 02/11/2023

import gymnasium as gym
import matplotlib.pyplot as plt
import time
import random
import numpy as np

RENDER_MODE = "text"  # choose between "graphic" or text;  graphic mode needs the pygame package to be installed
RENDER_FREQUENCY = 0.2  # output the game state at most every X seconds

env = gym.make('CliffWalking-v0', render_mode = "rgb_array" if RENDER_MODE == "graphic" else "ansi") # initialize the game

# This is a dumb agent that applies a policy that makes it run in circles
#   you can use this example agent class as a temple for implementing smarter ones
# Feel free to add methods if you consider it necessary
class DumbAgent():
    def __init__(self):
        self.previous_action = -1 # This agent needs to remember which direction he was going before so he can make nice circles

    # decide what action to take in the provided state by applying a certain policy
    def select_action(self, state):
        new_action = (self.previous_action + 1) % 4 # this agents policy is to run in circles
        self.previous_action = new_action
        return new_action
    
    # do the learning (e.g. update the Q-values, or collect data to do off-policy learning)
    def update(self, old_state, action, reward, new_state):
        pass # this agent just does not learn anything \_o_/

    # reset the agent to its initial state after an epoch. Can also be used to perform learning after an epoch)
    def reset(self, state):
        self.previous_action = -1

def to_coord(state_id):
    # we return the (x, y) coorinates, in the description page the use [y, x] to describe the locations
    return (state_id % 12, state_id // 12)

actions = {
    0: "up",
    1: "right",
    2: "down",
    3: "left"
}


def display_game(env, wait_time_s = RENDER_FREQUENCY):
    if RENDER_MODE == "graphic":
        img = env.render()  # show the current state of the game (the environment)
        plt.ion()
        if display_game.plt_im is None:
            display_game.plt_im = plt.imshow(img)
        else:
            display_game.plt_im.set_data(img)
        plt.draw()
        plt.show()
        plt.pause(0.001)
    else:
        img_ansi = env.render()  # show the current state of the game (the environment)
        print(img_ansi)
    time.sleep(wait_time_s)
display_game.plt_im = None


## Returns the number of steps taken and the ending reason (-1 if fallen off, 0 if survived but out of steps, 1 if reached goal)
def run_episode(agent, max_steps = 1000, muted = False):
    observation, info = env.reset() # restart the game
    agent.reset(observation) # reset the agent to its initial state
    if not muted:
        print(f"Starting in position: {to_coord(observation)}")

    for k in range(max_steps):
        # HINT: you may want to disable displaying of the game to run the experiments
        if not muted:
            display_game(env)

        action = agent.select_action(observation)  # select an action based on the current state
        new_observation, reward, terminated, truncated, info = env.step(action)  # perform the action and observe what happens 
        fallen_off_cliff = (reward == -100)  # Beware! we cannot check for cliff state because the environment automatically returns us to the starting position when falling off a cliff
        goal_reached = terminated  # if we reach the goal state, the environment returns terminated = True 

        agent.update(observation, action, reward, new_observation)  # perform some learning if the agent is capable of it

        if not muted:
            print(f"Action determined by agent: {actions[action]}")
            print(f"Reward for action: {actions[action]} in state: {observation} is: {reward}")
            print(f"New state is: {new_observation}")

        if goal_reached:
            if not muted:
                print(f"Goal reached after terminated after {k+1} steps\n\n")
            return k+1, 1  # we reached the goal
        elif fallen_off_cliff:
            if not muted:
                print(f"Fell off the cliff after {k+1} steps\n\n")
            
            return k+1, -1  # we fell off the cliff

        observation = new_observation

    if not muted:
        print(f"Survived for {k+1} steps but goal not reached\n\n")
    return k+1, 0  # we survived but did not reach the goal



def run_experiment(agent, episodes = 500):
    win_count = 0
    mute_output = False # you may want to mute the output if you run a lot of episodes

    for _ in range(episodes):
        steps, reason = run_episode(agent, muted=mute_output)
        if reason == 1:
            win_count += 1
    print(f"Reached goal {win_count} times out of {episodes} games")


if __name__ == "__main__":
    dumb_agent = DumbAgent()
    run_experiment(dumb_agent)

env.close() # end the game