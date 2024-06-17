import threading
import time

import numpy as np
from scipy.interpolate import CubicSpline

from air_hockey_challenge.framework.agent_base import AgentBase
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer, PuckTracker
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name
import mujoco
import torch
import matplotlib.pyplot as plt
import pickle
import torch
import random
import cv2
import os


import random
import numpy as np
import torch
from mp_pytorch.mp import MPFactory
from nmp import get_data_loaders_and_normalizer
from nmp import util
from nmp.aggregator import AggregatorFactory
from nmp.data_process import NormProcess
from nmp.decoder import DecoderFactory
from nmp.encoder import EncoderFactory
from nmp.net import MPNet

import yaml
import sys
import os

def save_video(images, filename, fps=30, size=None):
    """
    Save a series of NumPy array images as a video.

    :param images: List of NumPy arrays representing the images.
    :param filename: Output video file name.
    :param fps: Frames per second of the output video.
    :param size: Size of each frame (width, height). If None, the size of the first image is used.
    """
    # Determine the size of images if not provided
    if size is None:
        height, width = images[0].shape[:2]
        size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' if .avi file format is desired
    out = cv2.VideoWriter(filename, fourcc, fps, size)

    # Write the images to video file
    for img in images:
        # Ensure the image size matches the video size
        resized_img = cv2.resize(img, size)
        out.write(resized_img)

    # Release the VideoWriter object
    out.release()

def plot_traj_cartesian(traj, savepath=None):
    # plot the predicted puck trajectory
    plt.figure()
    plt.xlim(0, 3)
    plt.ylim(-0.52, 0.52)
    path_length = len(traj)
    colors = plt.cm.jet(np.linspace(0,1,path_length))
    plt.scatter(traj[:,0], traj[:,1], c=colors, zorder=20, s=10)
    
    # Create a ScalarMappable object using the colormap
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=path_length))
    sm.set_array([])  # Set the array to be empty, as we don't have actual data values

    # Add a colorbar with a legend
    plt.colorbar(sm, label='Path Length')
    if savepath is not None:
        plt.savefig(savepath)


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

class ProDMPAgent(AgentBase):
    def __init__(self, env_info, params, agent_id=1, seed=1, render=False,
                 model_path=None, model_epoch=10000, replan_horizon=10,
                 sample_batch_size=10, **kwargs):
        super(ProDMPAgent, self).__init__(env_info, agent_id, **kwargs)
        util.use_cuda()
        self.params = params
        self.seed = seed

        # initialize puck tracker
        self.puck_tracker = PuckTracker(env_info, agent_id=agent_id)

        # Get normalizer from dataset information
        dataset = util.load_pkl_dataset(params["dataset"]["task"],
                                        params["dataset"]["dataset_name"],
                                        params["dataset"]["max_path_length"],
                                        params["dataset"]["horizon"],
                                        params["mp"]["mp_args"]["dt"])
        _, _, _, self.normalizer \
            = get_data_loaders_and_normalizer(dataset, **params["dataset"], seed=self.seed)

        # initialize NN-MP
        self.encoder = EncoderFactory.get_encoders(**params["encoders"])
        self.aggregator = AggregatorFactory.get_aggregator(params["aggregator"]["type"], **params["aggregator"]["args"])
        self.decoder = DecoderFactory.get_decoder(params["decoder"]["type"], **params["decoder"]["args"])
        self.mpnet = MPNet(self.encoder, self.aggregator, self.decoder)
        self.mpnet.load_weights(*model_path, epoch=model_epoch)

        # initialize MP
        self.mp = MPFactory.init_mp(device="cuda", **params["mp"])

        # bookkeeping information for the agent
        self.dt = params["mp"]["mp_args"]["dt"]
        self.max_ctx_len = 4#params["assign_config"]["num_ctx_max"]
        self.replan_horizon = replan_horizon
        self.sample_batch_size = sample_batch_size

        self.reset()

    def reset(self):
        self.plan_idx = 0
        self.curr_ep_time = 0.0
        self.prev_obs_time = 0.0

        self.ctx_buffer = {"obs": {"time": [], "value": []}}  # up to 10 previous obs for inference
        self.action_buffer = []

    def draw_action(self, obs):
        self._add_obs_to_ctx(obs)
        if self.plan_idx == 0:
            self._sample_mp(obs)

        action = self.action_buffer[self.plan_idx]
        self.plan_idx += 1
        if self.plan_idx >= self.replan_horizon:
            self.plan_idx = 0

        self.curr_ep_time += self.dt

        return action

    def _add_obs_to_ctx(self, obs):
        if len(self.ctx_buffer["obs"]["time"]) >= self.max_ctx_len:
            self.ctx_buffer["obs"]["time"].pop(0)
            self.ctx_buffer["obs"]["value"].pop(0)

        self.ctx_buffer["obs"]["time"].append(self.curr_ep_time)
        self.ctx_buffer["obs"]["value"].append(obs)

    def _sample_mp(self, obs, savepath = "ee_traj.png", render=False):
        # process obs into encoder input

        # batch sample
        obs = torch.from_numpy(obs).repeat(self.sample_batch_size, 1).to("cuda")
        ctx_times = (torch.cuda.FloatTensor(self.ctx_buffer["obs"]["time"])[:, None]-self.curr_ep_time).repeat(self.sample_batch_size,1,1)
        ctx_values = torch.cuda.FloatTensor(self.ctx_buffer["obs"]["value"])[None, ...].repeat(self.sample_batch_size,1,1)

        norm_ctx_dict = NormProcess.batch_normalize(self.normalizer,
                                                    {"states":
                                                         {"time": ctx_times,
                                                          "value": ctx_values}})
        ctx_values = norm_ctx_dict["states"]["value"]

        ctx = {"ctx": torch.cat([ctx_times, ctx_values], dim=-1)}

        # Predict mean and Cholesky decomp of the weights distribution
        num_traj = self.sample_batch_size
        mean, diag, off_diag = self.mpnet.predict(num_traj=num_traj, enc_inputs=ctx, dec_input=None)

        # Compute trajectory distribution
        L = util.build_lower_matrix(diag, off_diag)
        mean = mean.squeeze(-2)
        L = L.squeeze(-3)
        assert mean.ndim == 3

        pred_time = (torch.arange(start=0, end=30)*self.dt).to("cuda")
        num_pred_time = 1
        num_agg = 1

        init_time = torch.zeros([num_traj])
        init_time = util.add_expand_dim(init_time, [1],
                                        [num_agg])

        robot_pos_vel_start_idx = 6
        init_pos = obs[:, robot_pos_vel_start_idx:robot_pos_vel_start_idx+self.mp.num_dof]
        init_pos = util.add_expand_dim(init_pos, [1], [num_agg])

        init_vel = obs[:, robot_pos_vel_start_idx+self.mp.num_dof:]
        init_vel = util.add_expand_dim(init_vel, [1], [num_agg])

        times = util.add_expand_dim(
            (pred_time.repeat(num_traj, 1)),
            add_dim_indices=[1], add_dim_sizes=[num_agg])

        # Reconstruct predicted trajectories
        self.mp.update_inputs(times=times, params=mean, params_L=L,
                         init_time=init_time, init_pos=init_pos,
                         init_vel=init_vel)
        traj_pos_mean = self.mp.get_traj_pos(flat_shape=False)
        traj_vel_mean = self.mp.get_traj_vel(flat_shape=False)
        # traj_pos_L = torch.linalg.cholesky(self.mp.get_traj_pos_cov())

        self.action_buffer = torch.stack([traj_pos_mean[0,0], traj_vel_mean[0,0]], dim=-2).detach().cpu().numpy()

    def get_joint_pos(self, obs):
        """
        Get the joint position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint position of the robot

        """
        return obs[self.env_info['joint_pos_ids']]

    def init_puck_tracker(self, obs):
        self.puck_tracker.reset(self.get_puck_pos(obs))

    def update_puck_tracker(self, obs):
        self.puck_tracker.step(self.get_puck_pos(obs))
        states, ps, t_predict = self.puck_tracker.get_prediction_trajectory(1.0)
        return states, ps



class DiffuserAgent(AgentBase):
    def __init__(self, env_info, diffusion_path, agent_id=1, n_timesteps = None, render=False, dataset_path = '/coc/data/sye40/air_hockey/data', **kwargs):
        super(DiffuserAgent, self).__init__(env_info, agent_id, **kwargs)
        args = Args()
        args.loadpath = diffusion_path

        trainer, dataset = utils.load_model(args.loadpath, 
                                          epoch=args.diffusion_epoch, 
                                        #   dataset_path='/home/sean/hockey_datasets/',
                                          dataset_path=dataset_path,
                                          )
        # self.dataset = diffusion_experiment.dataset
        # self.model = diffusion_experiment.trainer.ema_model
        
        self.dataset = dataset
        self.model = trainer.ema_model
        
        if n_timesteps is not None:
            # reduce the number of diffusion timesteps to increase sampling speed
            self.model.set_n_timesteps(n_timesteps)
            self.model.to(Args.device)


        self.puck_tracker = PuckTracker(env_info, agent_id=agent_id)

        self.min_x = 0
        self.max_x = 3
        self.min_y = -0.52
        self.max_y = 0.52

    def get_joint_pos(self, obs):
        """
        Get the joint position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint position of the robot

        """
        return obs[self.env_info['joint_pos_ids']]

    def init_puck_tracker(self, obs):
        self.puck_tracker.reset(self.get_puck_pos(obs))

    def update_puck_tracker(self, obs):
        self.puck_tracker.step(self.get_puck_pos(obs))
        states, ps, t_predict = self.puck_tracker.get_prediction_trajectory(1.0)
        return states, ps

    def compute_ee_grad(self, qpos, desired_pos, normalized_qpos=False):
        """
            normalized_qpos: if True, qpos input is expected normalized between -1 and 1 and we normalize the output as well
        """
        # self.robot_data.qpos = self.get_joint_pos(obs)

        if normalized_qpos:
            # need to unnormalize the qpos from the model
            qpos = ((qpos + 1) / 2) * (self.dataset.joint_pos_limit[1] - self.dataset.joint_pos_limit[0]) + self.dataset.joint_pos_limit[0]

        self.robot_data.qpos = qpos
        curr_pos = forward_kinematics(self.robot_model, self.robot_data, self.robot_data.qpos)[0]
        err_pos = desired_pos - curr_pos
        damp = 1e-3

        # j = jacobian(self.robot_model, self.robot_data, self.robot_data.qpos)

        name = link_to_xml_name(self.robot_model, 'ee')
        dtype = self.robot_data.qpos.dtype
        jac_pos = np.empty((3, self.robot_model.nv), dtype=dtype)
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, self.robot_model.body(name).id)

        update_joints = jac_pos.T @ np.linalg.inv(jac_pos @ jac_pos.T + damp * np.eye(3)) @ err_pos

        if normalized_qpos:
            update_joints = ((update_joints - self.dataset.joint_pos_limit[0]) / (self.dataset.joint_pos_limit[1] - self.dataset.joint_pos_limit[0])) * 2 - 1

        return update_joints
    
    def sample_actions(self, obs, puck_pred_states, savepath = "ee_traj.png", render=False):
        
        obs = ((obs - self.dataset.min_states) / (self.dataset.max_states - self.dataset.min_states))* 2 - 1
        puck_pos = obs[0:3]
        robot_pos = obs[6:9]
        # conditions = torch.tensor([0.0, 0.0])

        conditions = torch.tensor(puck_pred_states[:, :2])
        conditions = ((conditions - self.dataset.min_puck) / (self.dataset.max_puck - self.dataset.min_puck)) * 2 - 1
        conditions = conditions.unsqueeze(0) # add batch dimension

        conditions_dict = {'global_cond': conditions.float().to(Args.device)}
        local_cond = [[[], []]]

        start_time = time.time()
        # compute_grad_func = lambda desired_pos: self.compute_ee_grad(obs, desired_pos)
        # samples = to_np(self.model.conditional_sample(conditions_dict, 
        #                                               cond=local_cond, 
        #                                               ee_grad_func=self.compute_ee_grad, 
        #                                               sample_type="ee_cost", 
        #                                               puck_pred_states = puck_pred_states,))
        
        # samples = to_np(self.model.conditional_sample(conditions_dict, 
        #                                         cond=local_cond, 
        #                                         ee_grad_func=self.compute_ee_grad, 
        #                                         sample_type="original", 
        #                                         puck_pred_states = puck_pred_states,))
        
        samples = to_np(self.model.conditional_sample(conditions_dict, 
                                                      cond=local_cond, 
                                                    #   ee_grad_func=self.fk.end_effector,
                                                      sample_type = "MCG",  
                                                    #   sample_type="original", 
                                                    #   sample_type="ee_cost", 
                                                      puck_pred_states = puck_pred_states,))

        end_time = time.time()

        # print(f"Time taken: {end_time - start_time} seconds")

        # states_norm = samples[:, :, :12]
        # actions_norm = samples[:, :60, 12:]
        # actions_norm = samples[:, :, 12:]
        # states = ((states_norm + 1) / 2) * (dataset.max_states - dataset.min_states) + dataset.min_states
        actions = ((samples + 1) / 2) * (self.dataset.max_actions - self.dataset.min_actions) + self.dataset.min_actions
        actions = actions.reshape((1,-1, 2, 3))

        if render:
            self.plot_traj(actions, savepath = savepath)

        ### Extend actions
        # max_ep_len = env.info.horizon/2
        max_ep_len = 252
        extend_len = int(max_ep_len - actions.shape[1])
        act_extend = np.tile(actions[:, -1][:, None], (1,extend_len,1,1))
        actions = np.concatenate((actions, act_extend), axis=1)

        return actions
    
    def plot_traj(self, actions, savepath = "ee_traj.png"):
        # use forward kinematics to get ee position of the actions
        joint_angles = actions[0, :, 0, :]
        ee_pos = []
        for i in range(joint_angles.shape[0]):
            self.robot_data.qpos = joint_angles[i]
            ee_pos.append(forward_kinematics(self.robot_model, self.robot_data, self.robot_data.qpos)[0])

        ee_pos = np.array(ee_pos)
        # plt.figure()
        plt.xlim(self.min_x, self.max_x)
        plt.ylim(self.min_y, self.max_y)
        path_length = len(joint_angles)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.scatter(ee_pos[:,0], ee_pos[:,1], zorder=20, s=1, label="Diffusion")
        
        # Create a ScalarMappable object using the colormap
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=path_length))
        # sm.set_array([])  # Set the array to be empty, as we don't have actual data values

        # Add a colorbar with a legend
        # plt.colorbar(sm, label='Path Length')
        plt.savefig(savepath)
        plt.close()

def success_defend(current_traj, agent):
    mallet_radius = agent.env_info['mallet']['radius']
    current_traj = np.vstack(current_traj)
    puck_radius = 0.03165
    joint_angles = current_traj[:, 6:9]
    ee_pos = []
    for i in range(joint_angles.shape[0]):
        agent.robot_data.qpos = joint_angles[i]
        ee_pos.append(forward_kinematics(agent.robot_model, agent.robot_data, agent.robot_data.qpos)[0])
    ee_pos = np.array(ee_pos)

    puck_trajectory = current_traj[:, :2]

    # for every point in the puck trajectory, check if it the distance to the ee is less than the sum of the radii
    # do batched operation
    ee_pos = ee_pos[:, :2]
    dist = np.linalg.norm(ee_pos - puck_trajectory, axis=-1)
    # print(dist)
    # print(puck_radius + mallet_radius)
    # print(np.any(dist < puck_radius + mallet_radius))
    # add another small margin
    return np.any(dist < puck_radius + mallet_radius - 0.005), ee_pos

def save_success_rate(successes, total_number, diffusion_timesteps, folderpath):
    filepath = f"{folderpath}/results.txt"
    # check if file does not exist
    with open(filepath, 'a') as file:
        file.write(f"D-timesteps: {diffusion_timesteps}: Success {successes} out of {total_number}, {successes/total_number} \n")
    file.close()
    print("Saved to ", filepath)

def main():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("tkAgg")

    util.use_cuda()

    if len(sys.argv) != 2:
        exit("Expected format: python sample_model_test.py [path/to/model]")

    exp_dir = sys.argv[1]
    config_yml = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if "relative" in f]
    model_dir = [os.path.join(exp_dir, f, "log/rep_00/model") for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f))]

    config = []
    with open(*config_yml) as stream:
        full_config = yaml.safe_load_all(stream)
        for doc in full_config:
            config.append(doc)
    params = config[1]['params']

    render = False
    render_video = False
    # set seeds
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    successes = 0
    total = 1000

    env = AirHockeyChallengeWrapper(env="3dof-hit", interpolation_order=3, debug=False)
    
    render_path = "videos/hard_v2"
    # check if the directory exists
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    successes = 0

    agent = ProDMPAgent(env.base_env.env_info,
                        params=params,
                        seed=seed,
                        model_path=model_dir,
                        model_epoch=10000,
                        replan_horizon=10)

    # sample a starting configuration from the data distribution
    dataset_path = os.path.join("nmp/dataset", params["dataset"]["task"], params["dataset"]["dataset_name"]) + '.pkl'
    traj_list = []
    load_demo_data(traj_list, dataset_path, num_traj_to_load=100)
    demo_init_state, demo_actions, preference = sample_demo_data(traj_list)

    for n in range(total):
        current_traj = []
        obs = env.reset(demo_init_state.copy())
        current_traj.append(obs.copy())
        agent.init_puck_tracker(obs)
        agent.update_puck_tracker(obs)
        # first actions should be empty to get the puck trajectory

        jp = agent.get_joint_pos(obs)
        action = np.zeros((2, 3))
        action[0] = jp
        obs, reward, done, info = env.step(action)
        current_traj.append(obs.copy())
        puck_pred_states, _ = agent.update_puck_tracker(obs)

        if render:
            plot_traj_cartesian(puck_pred_states)

        # actions = agent.sample_actions(obs, puck_pred_states, savepath=f"{render_path}/traj_{n}.png", render=render)

        steps = 0
        num_trajs = 0
        trajs_needed = 20
        trajs = []
        current_traj = []
        frames = []

        while not done and steps < env.info.horizon / 2:
            steps += 1

            action = agent.draw_action(obs)
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            env.render(record=False)
            if render_video:
                img = env.render(record=True)
                frames.append(img)

        print("Collected Trajectory: ", num_trajs)
        print(len(frames))

        hit, ee_pos = success_defend(current_traj, agent)
        successes += hit

        print("Hit: ", hit)

        if render:
            current_traj = np.vstack(current_traj)
            # let's also plot the true trajectory
            plt.figure(figsize=[5,5])
            points_whole_ax = 5 * 0.8 * 72
            puck_radius = 0.03165
            puck_radius_points = 2 * puck_radius / 1.0 * points_whole_ax
            mallet_radius = 0.04815
            mallet_radius_points = 2 * mallet_radius / 1.0 * points_whole_ax
            plt.xlim(0, 3)
            plt.ylim(-0.52, 0.52)
            # plt.scatter(current_traj[:,0], current_traj[:,1], zorder=20, s=puck_radius_points**2) # puck trajectory
            plt.scatter(current_traj[:,0], current_traj[:,1], zorder=20, s=3) # puck trajectory
            joint_angles = current_traj[:, 6:9]
            # plt.scatter(ee_pos[:,0], ee_pos[:,1], zorder=20, s=mallet_radius_points**2, label="EE path")
            plt.scatter(ee_pos[:,0], ee_pos[:,1], zorder=20, s=4, label="EE path")
            plt.savefig(f"{render_path}/traj_{n}_true.png")
        
        if render_video:
            save_video(frames, f"{render_path}/{n}.mp4", fps=30, size=(1600,1200))
            # trajs.append(current_traj.copy())
        steps = 0
        current_traj = []
        obs = env.reset()

    save_success_rate(successes, total, 100, loadpath)
    # if render:
    #     save_video(frames, f"videos/defender/all_trajs.mp4", fps=30, size=(1600,1200))

if __name__ == '__main__':
    main()