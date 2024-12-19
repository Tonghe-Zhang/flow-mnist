import os
import sys

from datetime import datetime
from tqdm import tqdm as tqdm
import torch



os.environ['PYTHONPATH']=os.getcwd()
# Get the directory of the current script
'''
export PYTHONPATH=$PYTHONPATH:/home/zhangtonghe/flow-mnist/data
'''
current_dir = os.path.dirname(os.path.abspath(__file__))
iprt_dirs=[]
iprt_dirs.append(os.path.join(current_dir, '..', 'model'))
iprt_dirs.append(os.path.join(current_dir, '..', 'data'))
for import_dir in iprt_dirs:
    if import_dir not in sys.path:
        sys.path.append(import_dir)

from model import *

from data.mnist_dataset import *

# enable hydra full error:
os.environ['HYDRA_FULL_ERROR'] = '1'
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from helpers import visualize


# config_dict = {
#     "device": "cuda:7",
#     "data_shape": [1, 28, 28],
#     "batch_size": 1024,
#     "data_hidden_dim": 64,
#     "cls_hidden_dim": 32,
#     "time_hidden_dim": 64,
#     "algo": {
#         "_target_": "alg.reflow.ReFlow",
#         "device": "cuda:7",
#         "data_shape": [1, 28, 28],
#         "model": {
#             "_target_": "model.flow_mlp.FlowMLP",
#             "flow_mlp_cfg": {
#                 "hidden_dims": [128, 256, 128],
#                 "output_dim": 784,
#                 "time_hidden_dim": 64,
#                 "cls_hidden_dim": 32,
#                 "data_hidden_dim": 64,
#             },
#         },
#         "train_cfg": {
#             "n_epochs": 300,
#             "eval_interval": 20,
#             "inference_steps": 64,
#             "lr": 1e-3,
#             "warmup_epochs": 10,
#             "max_epochs": 400,
#             "min_lr": 1e-6,
#             "warmup_start_lr": 1e-5,
#             "batch_size": 1024,
#         },
#     },
#     "dataset": {
#         "device": "cuda:7",
#         "batch_size": 1024,
#         "use_first_train": -1,
#         "use_first_eval": -1,
#     },
# }


import ray 
# ray.init(num_gpus=8)

ray.init(num_gpus=8)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def run_one(config, checkpoint_dir=None):
    from alg import ReFlow
    from model import FlowMLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from dataclasses import dataclass
    from typing import List
    @dataclass
    class FlowMLPCfg:
        hidden_dims: List[int]
        output_dim: int
        time_hidden_dim: int
        cls_hidden_dim: int
        data_hidden_dim: int
        batch_norm: bool
    
    flow_mlp_cfg = FlowMLPCfg(
        hidden_dims=[128, 256, 128],
        output_dim=784,
        data_hidden_dim=512,
        cls_hidden_dim=256,
        time_hidden_dim=256,
        # time_hidden_dim=64,
        # cls_hidden_dim=32,
        # data_hidden_dim=64,
        batch_norm=True
    )
    
    @dataclass
    class TrainCfg:
        n_epochs: int
        eval_interval: int
        inference_steps: int
        lr: float
        max_epochs: int
        batch_size: int
        use_bc_loss: bool
    
    train_cfg = TrainCfg(
        n_epochs=400,
        eval_interval=20,
        inference_steps=128,
        lr=config["lr"],
        max_epochs=400,
        batch_size=config["batch_size"],
        use_bc_loss=bool(config["use_bc_loss"])
    )

    reflow = ReFlow(device=device,
                    data_shape=[1, 28, 28],
                    model=FlowMLP(flow_mlp_cfg),
                    train_cfg=train_cfg)
    
    reflow.network.to(reflow.device)
    print(f"relow={reflow}")
    
    train_dataset = MNISTDataset(device = device, csv_file='/home/zhangtonghe/flow-mnist/data/mnist_train.csv', use_top=-1)
    test_dataset = MNISTDataset(device = device, csv_file='/home/zhangtonghe/flow-mnist/data/mnist_test.csv', use_top=-1)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, drop_last=True)
    
    train_losses = []
    eval_losses = []
    best_eval_loss = float('inf')
    
    for epoch in range(1, train_cfg.n_epochs + 1):
        reflow.network.train()
        epoch_train_loss = []
        for (x, cls) in train_loader:
            x.to(reflow.device)
            cls.to(reflow.device)

            (xt, t, cls), v = reflow.generate_target(x, cls)

            reflow.optimizer.zero_grad()

            loss = reflow.loss(xt, t, cls, v)
            
            if reflow.use_bc_loss==True:
                alpha=0.5
                loss += alpha* torch.nn.functional.mse_loss(xt,x)
            
            epoch_train_loss.append(loss.item())

            loss.backward()
            reflow.optimizer.step()
        # Calculate average training loss for this epoch
        train_loss_mean = np.mean(epoch_train_loss)
        train_loss_std = np.std(epoch_train_loss)
        train_losses.append((train_loss_mean, train_loss_std))
        train_epochs = np.arange(start=1, stop=epoch + 1, step=1)

        if epoch % reflow.eval_interval == 0:
            # evaluate model
            err_list = []
            eval_loss_list = []
            reflow.network.eval()
            with torch.no_grad():
                
                for (x, cls) in test_loader:
                    x.to(reflow.device)
                    cls.to(reflow.device)
                    x_hat = reflow.sample(cls=cls, inference_steps=reflow.inference_steps, record_intermediate=False)

                    mse_err = F.mse_loss(x_hat, x)
                    err_list.append(mse_err.item())

                    eval_targets, eval_v = reflow.generate_target(x, cls)
                    eval_loss_list.append(reflow.loss(*eval_targets, eval_v).item())

            eval_loss_mean = np.mean(eval_loss_list)
            eval_loss_std = np.std(eval_loss_list)
            eval_losses.append((eval_loss_mean, eval_loss_std))     
            
            import ray
            ray.train.report(metrics={"eval_loss":eval_loss_mean},
                             checkpoint=None)
            
            if eval_loss_mean < best_eval_loss:
                best_eval_loss = eval_loss_mean
        
        reflow.scheduler.step()  # Update the learning rate scheduler

if __name__ == "__main__":
    # Hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-5, 1e-2), #tune.choice([1e-4, 1e-3, 3*1e-3, 5*1e-3, 8*1e-3]), #tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([128, 256, 512]),
        "use_bc_loss": tune.choice([1, 0])
    }
    scheduler = ASHAScheduler(metric="eval_loss", mode="min", max_t=300, grace_period=10, reduction_factor=2)  # Reduce the number of trials by half after each evaluation)
    reporter = CLIReporter(metric_columns=["eval_loss"])
    
    result = tune.run(  
        run_one,
        resources_per_trial={"cpu": 3.2, "gpu":0.25}, # Adjust based on your resources
        max_concurrent_trials = 40,
        config=config,
        num_samples=80,
        scheduler=scheduler,
        progress_reporter=reporter,
        # checkpoint_freq=100,  # Save checkpoints every 50 epochs
        # checkpoint_at_end=True,  # Only save the last checkpoint
        local_dir="/home/zhangtonghe/flow-mnist/ray_tune" # Directory for saving results
    )