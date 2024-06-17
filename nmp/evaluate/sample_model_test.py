import random
import numpy as np
import torch
from mp_pytorch.mp import MPFactory
from nmp import get_data_loaders_and_normalizer
from nmp import nll_loss
from nmp import util
from nmp.aggregator import AggregatorFactory
from nmp.data_process import NormProcess
from nmp.decoder import DecoderFactory
from nmp.encoder import EncoderFactory
from nmp.net import MPNet
from nmp.net import avg_batch_loss


import yaml
import sys
import os


if __name__ == "__main__":
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

    # set seeds
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize NN-MP
    encoder = EncoderFactory.get_encoders(**params["encoders"])
    aggregator = AggregatorFactory.get_aggregator(params["aggregator"]["type"], **params["aggregator"]["args"])
    decoder = DecoderFactory.get_decoder(params["decoder"]["type"], **params["decoder"]["args"])
    mpnet = MPNet(encoder, aggregator, decoder)
    mpnet.load_weights(*model_dir, epoch=10000)

    # initialize MP
    mp = MPFactory.init_mp(device="cuda", **params["mp"])

    # Test sampling a trajectory based on observations in the dataset
    dataset = util.load_npz_dataset(params["dataset"]["name"])
    train_loader, valid_loader, test_loader, normalizer \
        = get_data_loaders_and_normalizer(dataset, **params["dataset"], seed=seed)
    batch = next(iter(test_loader))

    # Get encoder input (single timestep)
    time_ctx_last = batch["box_robot_state"]["time"][:, 0]
    ctx_times = (batch["box_robot_state"]["time"]
                 - time_ctx_last[:, None])[:, 0][..., None, None]
    ctx_values = batch["box_robot_state"]["value"][:, 0][:, None, :]
    norm_ctx_dict = NormProcess.batch_normalize(normalizer,
                                                {"box_robot_state":
                                                     {"time": ctx_times,
                                                      "value": ctx_values}})
    ctx_values = norm_ctx_dict["box_robot_state"]["value"]

    ctx = {"ctx": torch.cat([ctx_times, ctx_values], dim=-1)}

    # Predict mean and Cholesky decomp of the weights distribution
    num_traj = batch["box_robot_state"]["value"].shape[0]
    mean, diag, off_diag = mpnet.predict(num_traj=num_traj, enc_inputs=ctx, dec_input=None)

    # Compute trajectory distribution
    L = util.build_lower_matrix(diag, off_diag)
    mean = mean.squeeze(-2)
    L = L.squeeze(-3)
    assert mean.ndim == 3

    num_pred_pairs = 1
    pred_pairs = torch.arange(start=0, end=50)
    num_agg = 1

    init_time = torch.zeros([num_traj])
    init_time = util.add_expand_dim(init_time, [1],
                                    [num_agg])
    init_pos = batch["des_cart_pos_vel"]["value"][:, 0, :mp.num_dof]
    init_pos = util.add_expand_dim(init_pos, [1], [num_agg])

    init_vel = batch["des_cart_pos_vel"]["value"][:, 0, mp.num_dof:]
    init_vel = util.add_expand_dim(init_vel, [1], [num_agg])

    times = util.add_expand_dim(
        (batch["des_cart_pos_vel"]["time"] - time_ctx_last[:, None])[:,
        pred_pairs],
        add_dim_indices=[1], add_dim_sizes=[num_agg])

    # Reconstruct predicted trajectories
    mp.update_inputs(times=times, params=mean, params_L=L,
                          init_time=init_time, init_pos=init_pos,
                          init_vel=init_vel)
    traj_pos_mean = mp.get_traj_pos(flat_shape=False).detach().cpu().numpy()
    traj_pos_L = torch.linalg.cholesky(mp.get_traj_pos_cov())

    util.debug_plot(x=None, y=[traj[0, :, 0] for traj in traj_pos_mean], title="ProDMP pos 1")
    util.debug_plot(x=None, y=[traj[:100, 0] for traj in batch["des_cart_pos_vel"]["value"]], title="GT pos 1")
    # util.debug_plot(x=None, y=[traj[0, :, 1] for traj in traj_pos_mean], title="ProDMP pos 2")



