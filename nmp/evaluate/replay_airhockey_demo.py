import time
import threading
import numpy as np
from scipy.interpolate import CubicSpline
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils import inverse_kinematics, world_to_robot
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer
import math
from scipy.optimize import fsolve
import pickle

import os
import sys

import yaml
import time
import numpy as np

import matplotlib.pyplot as plt
# import cloudpickle as pickle
import time

data_file = "/home/core/workspace/kml/ProDMP_RAL/nmp/dataset/airhockey_defend/straight_1000_pose_sv_w_preference_subset.pkl"
render = True
dt_ = 0.02

def load_data_from_pickle(datafile):
    with open(datafile, 'rb') as f:
        demo_states = pickle.load(f)

    return demo_states

def load_demo_data(traj_list, dataset_path, num_traj_to_load):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    traj_list.extend(data[:num_traj_to_load])

def sample_demo_data(traj_list):
    # traj_num = jax.random.randint(rng_key, (1,), 0, len(traj_list))

    # traj_data = traj_list[traj_num[0]]
    traj_data = traj_list[50]    # comment to not use hardcoded traj
    traj_init_state = traj_data[0][0]
    traj_actions = [traj_data[i][1] for i in range(len(traj_data))]
    pref = traj_data[0][6]

    return (traj_init_state, traj_actions, pref)  # list of (2,3)


def main():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    env = AirHockeyChallengeWrapper(env="3dof-hit", interpolation_order=3, debug=True)

    # Load preference from demo
    traj_list = []
    load_demo_data(traj_list, data_file, num_traj_to_load=100)
    demo_init_state, demo_actions, preference = sample_demo_data(traj_list)

    state = env.reset(demo_init_state.copy())

    n = 100000000
    successful_trajs = 0
    episodes_executed = 0

    steps = 0

    while(True):
        steps += 1
        # print(f"Step {i}")

        # Use below to execute a demo trajectory
        print(f"State: {state}")
        action = demo_actions[steps-1]
        # if i > 0:
        #     action[0] = demo_actions[i-1][0] + action[1]*dt_
        state, reward, done, info = env.step(action)

        if render:
            env.render(record=False)

        if done or steps > env.info.horizon / 2:
            episodes_executed += 1
            if info["success"] == 1:
                successful_trajs += 1
            steps = 0
            print(f"Current success rate: {successful_trajs}/{episodes_executed} ({(successful_trajs/episodes_executed)*100}%)")
            print(f"Episodes executed: {episodes_executed}")

            if episodes_executed == 1000:
                break

            state = env.reset(demo_init_state.copy())


    # Plot
    # num_pts = 100
    # dt = 0.02
    # x = np.linspace(0, num_pts * dt, num_pts)
    # for dof in range(len(action_buffer[0][0])):
    #     plt.plot(x, [action_buffer[i][0][dof] for i in range(num_pts)], label="command")
    #     plt.plot(x, [replay_buffer["state"][i][dof+6] for i in range(num_pts)], label="next_state")
    #     plt.legend()
    #     plt.show()

if __name__ == '__main__':
    main()
