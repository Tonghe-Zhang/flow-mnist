import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# safe import
import os
import sys
import torch.optim.optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model directory
model_dir = os.path.join(current_dir, '..', 'model')
# Add the model directory to sys.path
if model_dir not in sys.path:
    sys.path.append(model_dir)
from model import *

from data  import *
from script.helpers import *
import hydra
from omegaconf import OmegaConf, DictConfig
import time
from torch.optim.optimizer import *
from torch.optim.lr_scheduler import *


class ReFlow(nn.Module):
    
    def __init__(self, device, data_shape, model, train_cfg, model_path=None):
        super(ReFlow, self).__init__()
        
        
        self.train_cfg= train_cfg
        
        self.device=device
        
        self.data_shape=tuple(data_shape)
        
        self.network: nn.Module
        self.network = model.to(self.device)
        if model_path is not None:
            self.load_model(model_path)
            
        self.n_epochs= train_cfg.n_epochs
        
        self.eval_interval=train_cfg.eval_interval
        
        self.inference_steps = train_cfg.inference_steps
        
        self.use_bc_loss = train_cfg.use_bc_loss
        
        self.batch_size = train_cfg.batch_size
        
        self.lr = train_cfg.lr
        
        self.optimizer = torch.optim.AdamW(params=self.network.parameters(), lr=self.lr, weight_decay=1e-6)
        
        self.use_scheduler = train_cfg.use_scheduler
        if self.use_scheduler:
            self.scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=train_cfg.lr, total_steps=self.n_epochs)
        
        # self.scheduler = CosineAnnealingLR(optimizer=self.optimizer,
        #                                    T_max=self.n_epochs)
        # self.scheduler = CosineAnnealingWarmupLR(
        #             self.optimizer,
        #             warmup_epochs=train_cfg.warmup_epochs,
        #             max_epochs=train_cfg.max_epochs,
        #             min_lr=train_cfg.min_lr,
        #             warmup_start_lr=train_cfg.warmup_start_lr
        #             )
        
        print(f"ReFlow object contains self.network={self.network}")
    
    def generate_target(self, x1, cls):
        '''
        inputs:
            x1. tensor. real data
            cls. class label
        outputs:
            (xt, t, cls): tuple, the inputs to the model. containing...
                xt: corrupted data. torch.  torch.Size([N,C,H,W])
                t:  corruption ratio        torch.Size([N])
                cls. class label            torch.Size([N])
            v:  tensor. target velocity, from x0 to x1 (v=x1-x0). the desired output of the model. torch.Size([N, C, H, W])
        '''
        # random time, or mixture ratio between (0,1). different for each sample, but he same for each channel. 
        t=torch.randn(self.batch_size,device=self.device)
        t_broadcast=(torch.ones_like(x1, device=self.device) * t.view(self.batch_size, 1, 1, 1)).to(self.device)
        # generate random noise
        x0=torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        # generate corrupted data
        

        xt= t_broadcast* x1 + (1-t_broadcast)* x0
        # generate target
        v=x1-x0

        # print(f"xt.shape={xt.shape}")
        # print(f"t.shape={t.shape}")
        # print(f"cls.shape={cls.shape}")
        
        return (xt, t, cls), v
    
    @torch.no_grad()
    def sample(self, cls, inference_steps:int, sample_batch_size=None, record_intermediate=False):
        '''
        inputs:
            cls: label
            num_step: number of denoising steps in a single generation. 
            record_intermediate: whether to return predictions at each step
        outputs:
            if `record_intermediate` is False, xt. tensor of shape `self.data_shape`
            if `record_intermediate` is True,  xt_list. tensor of shape `[num_steps,self.data_shape]`
        '''
        if sample_batch_size is None:
            sample_batch_size=self.batch_size
        
        if record_intermediate:
            x_hat_list=torch.zeros((inference_steps,)+self.data_shape)  # [num_steps,self.data_shape]
        
        x_hat=torch.randn((sample_batch_size,)+self.data_shape, device=self.device)    # [batchsize, C, H, W]
        
        dt = (1/inference_steps)* torch.ones_like(x_hat).to(self.device)
        
        steps = torch.linspace(0,1,inference_steps).repeat(sample_batch_size, 1).to(self.device)                       # [batchsize, num_steps]
        
        for i in range(inference_steps):
            t = steps[:,i]
            vt=self.network(x_hat,t,cls)
            x_hat+= vt* dt            
            if record_intermediate:
                x_hat_list[i] = x_hat
        return x_hat_list if record_intermediate else x_hat

    def loss(self, xt, t, cls, v):
        
        v_hat = self.network(xt, t, cls)
        
        loss = F.mse_loss(input=v_hat, target=v)
        
        return loss 
    
    def generate_and_visualize(self, epoch, file_path):
        # Generate samples for each class
        fig, axes = plt.subplots(1, 10, figsize=(20, 4))
        fig.suptitle(f"Model Epoch: {epoch}", fontsize=16)

        for i, cls in enumerate(range(10)):
            cls = torch.tensor(cls).unsqueeze(0).to(self.device)
            x_hat = self.sample(cls, inference_steps=self.inference_steps, sample_batch_size=1, record_intermediate=False) #[self.data_shape] or [num_steps,self.data_shape]
            axes[i].imshow((x_hat[0,0].cpu()/2.0+0.5) * 225.0, cmap='gray')
            axes[i].set_title(f"Class {cls[0].item()}", fontsize=12)
            axes[i].axis('off')
        
        # Save the plot
        plt.savefig(file_path)
        plt.close(fig)
        print(f'''plot saved to {file_path}''')
    
    def generate_and_visualize_intermediate(self, fig_path):
        
        from tqdm import tqdm
        # Generate samples for each class with intermediate stages
        if self.inference_steps<5:
            intermediate_steps = list(range(self.inference_steps))
        else:
            intermediate_steps = [0, int(self.inference_steps * 0.2), int(self.inference_steps * 0.4), int(self.inference_steps * 0.6), int(self.inference_steps * 0.8), self.inference_steps - 1]
        num_intermediate = len(intermediate_steps)

        # Create a figure with 5 rows and 10 columns
        fig, axes = plt.subplots(num_intermediate, 10, figsize=(20, 10))
        fig.suptitle(f"ReFlow Generation", fontsize=16)

        for i, cls in tqdm(enumerate(range(10))):
            cls_tensor = torch.tensor(cls).unsqueeze(0).to(self.device)
            x_hat = self.sample(cls_tensor, inference_steps=self.inference_steps, sample_batch_size=1, record_intermediate=True)  # Get the intermediate results

            for j, step in tqdm(enumerate(intermediate_steps)):
                # Display the intermediate generated image for the current class at this step
                axes[j, i].imshow((x_hat[step, 0].cpu() / 2.0 + 0.5) * 225.0, cmap='gray')
                axes[j, i].set_title(f"Class {cls} - Step {step}", fontsize=8)
                axes[j, i].axis('off')

        # Adjust layout
        plt.tight_layout()  # Make room for the title  rect=[0, 0.03, 1, 0.95]
        plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between rows
        
        # Save the plot
        plt.savefig(fig_path)
        plt.close(fig)
        print(f'Plot saved to {fig_path}')
    
    def load_model(self,model_path:str):
        print(f"loading model from {model_path}")
        self.network.load_state_dict(torch.load(model_path,  weights_only=True))
        self.network.to(device=self.device)
        print(f"...loading complete.")
            
    def run(self, cfg, train_loader: DataLoader, test_loader: DataLoader):
        
        use_bc_loss = self.use_bc_loss
        
        train_losses = []
        if use_bc_loss:
            bc_losses = []
        eval_losses = []
        reconstruct_losses = []
        fid_scores = []

        log_dir = os.path.join(BASE_DIR, 'log', current_time())
        image_dir = os.path.join(log_dir, 'visualize')
        ckpt_dir = os.path.join(log_dir, 'ckpt')
        cfg_path = os.path.join(ckpt_dir, 'cfg.yaml')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        # dump the config file used to train. 
        OmegaConf.save(cfg, cfg_path)

        self.network.to(self.device)
        for epoch in tqdm(range(1, self.n_epochs + 1, 1)):
            start_time = time.time()  # Start time for the epoch

            self.network.train()
            epoch_train_loss = []
            if use_bc_loss:
                epoch_bc_loss = []

            for step, (x, cls) in tqdm(enumerate(train_loader),desc=f'epoch{epoch}/step:'):
                x.to(self.device)
                cls.to(self.device)

                (xt, t, cls), v = self.generate_target(x, cls)

                self.optimizer.zero_grad()

                loss = self.loss(xt, t, cls, v)
                if use_bc_loss:
                    bc_loss = F.mse_loss(xt, x)
                    loss+=bc_loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss.append(loss.item())
                if use_bc_loss:
                    epoch_bc_loss.append(bc_loss.item())
                
                # print(f"epoch: {epoch}, steps: {step}, loss: {loss.item():.4}", end="")

            # Calculate average training loss for this epoch
            train_loss_mean = np.mean(epoch_train_loss)
            train_loss_std = np.std(epoch_train_loss)
            train_losses.append((train_loss_mean, train_loss_std))
            train_epochs = np.arange(start=1, stop=epoch + 1, step=1)
            
            if use_bc_loss:
                bc_loss_mean = np.mean(epoch_bc_loss)
                bc_loss_std = np.std(epoch_bc_loss)
                bc_losses.append((bc_loss_mean, bc_loss_std))

            if epoch % self.eval_interval == 0:
                # evaluate model
                err_list = []
                eval_loss_list = []
                fid_list = []
                self.network.eval()
                with torch.no_grad():
                    
                    self.generate_and_visualize(epoch, file_path=os.path.join(ckpt_dir, f'model_{epoch}.png'))
                    
                    for (x, cls) in test_loader:
                        x.to(self.device)
                        cls.to(self.device)
                        x_hat = self.sample(cls=cls, inference_steps=self.inference_steps, record_intermediate=False)

                        mse_err = F.mse_loss(x_hat, x)
                        err_list.append(mse_err.item())

                        # fid = calculate_fid(x_hat.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1), device=self.device)
                        fid=0.0
                        fid_list.append(fid)

                        eval_targets, eval_v = self.generate_target(x, cls)
                        eval_loss_list.append(self.loss(*eval_targets, eval_v).item())

                if len(eval_loss_list) <=0:
                    raise ValueError("eval_loss_list must be a list")
                eval_loss_mean = np.mean(eval_loss_list) 
                eval_loss_std = np.std(eval_loss_list)
                eval_losses.append((eval_loss_mean, eval_loss_std))

                reconstruct_loss_mean = np.mean(err_list)
                reconstruct_loss_std = np.std(err_list)
                reconstruct_losses.append((reconstruct_loss_mean, reconstruct_loss_std))

                fid_mean = np.mean(fid_list)
                fid_std = np.std(fid_list)
                fid_scores.append((fid_mean, fid_std))

                eval_epochs = np.arange(start=self.eval_interval, stop=epoch + 1, step=self.eval_interval)

                # Calculate stats for logging
                epoch_duration = time.time() - start_time  # Duration of the epoch
                remaining_epochs = self.n_epochs - epoch
                estimated_remaining_time = remaining_epochs * epoch_duration

                # Print logs in mean ± std format
                if use_bc_loss==False:
                    print(f'Epoch [{epoch}/{self.n_epochs}], '
                        f'Train Loss: {train_loss_mean:.4f} ± {train_loss_std:.4f}, '
                        f'Eval Loss: {eval_loss_mean:.4f} ± {eval_loss_std:.4f}, '
                        f'Reconstruction: {reconstruct_loss_mean:.4f} ± {reconstruct_loss_std:.4f}, '
                        # f'FID score: {fid_mean:.4f} ± {fid_std:.4f}, '
                        f'Learning rate: {self.lr:4f}, '
                        f'Time: {epoch_duration:.2f}s, '
                        'Estimated Remaining Time:' + format_time_seconds(estimated_remaining_time))
                else:
                    print(f'Epoch [{epoch}/{self.n_epochs}], '
                        f'Flow Loss: {train_loss_mean:.4f} ± {train_loss_std:.4f}, '
                        f'Train BC Loss: {bc_loss_mean:.4f} ± {bc_loss_std:.4f}, '
                        f'Eval BC Loss: {eval_loss_mean:.4f} ± {eval_loss_std:.4f}, '
                        f'Reconstruction: {reconstruct_loss_mean:.4f} ± {reconstruct_loss_std:.4f}, '
                        # f'FID score: {fid_mean:.4f} ± {fid_std:.4f}, '
                        f'Learning rate: {self.lr:4f}, '
                        f'Time: {epoch_duration:.2f}s, '
                        'Estimated Remaining Time:' + format_time_seconds(estimated_remaining_time))

                # Plot the loss, eval curves with shaded variances
                plt.figure(figsize=(10, 5))
                train_means, train_stds = zip(*train_losses)
                if use_bc_loss:
                    bc_means, bc_stds = zip(*bc_losses)
                eval_means, eval_stds = zip(*eval_losses)
                reconstruct_means, reconstruct_stds = zip(*reconstruct_losses)
                fid_means, fid_stds = zip(*fid_scores)

                plt.plot(train_epochs, train_means, label='Train Loss', color='red')
                plt.fill_between(train_epochs,
                                np.array(train_means) - np.array(train_stds),
                                np.array(train_means) + np.array(train_stds),
                                color='red', alpha=0.2)
                
                if use_bc_loss:
                    plt.plot(train_epochs, bc_means, label='Train BC Loss', color='orange')
                    plt.fill_between(train_epochs,
                                np.array(bc_means) - np.array(bc_stds),
                                np.array(bc_means) + np.array(bc_stds),
                                color='orange', alpha=0.2)
                
                plt.plot(eval_epochs, eval_means, label='Eval Loss', color='black')
                plt.fill_between(eval_epochs,
                                np.array(eval_means) - np.array(eval_stds),
                                np.array(eval_means) + np.array(eval_stds),
                                color='black', alpha=0.2)

                plt.plot(eval_epochs, reconstruct_means, label='Reconstruct Error', color='blue')
                plt.fill_between(eval_epochs,
                                np.array(reconstruct_means) - np.array(reconstruct_stds),
                                np.array(reconstruct_means) + np.array(reconstruct_stds),
                                color='blue', alpha=0.2)

                # plt.plot(eval_epochs, fid_means, label='FID', color='purple')
                # plt.fill_between(eval_epochs,
                #                 np.array(fid_means) - np.array(fid_stds),
                #                 np.array(fid_means) + np.array(fid_stds),
                #                 color='purple', alpha=0.2)
                plt.title('Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(image_dir, 'loss.png'))
                print(f"loss curves saved to {os.path.join(image_dir, 'loss.png')}")
                plt.close()  # Close the figure to free up memory
                # Save model after evaluation
                
                
                
                # checkpoint = {
                #     'model_state_dict': self.network.state_dict(),  # Save model weights
                #     'optimizer_state_dict': self.optimizer.state_dict(),  # Save optimizer state
                #     'epoch': epoch,  # Current epoch
                #     }
                torch.save(self.network.state_dict(), os.path.join(ckpt_dir, f'model_{epoch}.pth'))
            
            # if self.epoch % self.update_ema_freq == 0:
            #     self.step_ema()
            if self.use_scheduler:
                self.scheduler.step()  # Update the learning rate scheduler
                self.lr = self.scheduler.get_last_lr()[0]
            else:
                self.lr = self.optimizer.param_groups[0]['lr'] # or: for multiple parameter groups. [param_group['lr'] for param_group in optimizer.param_groups]
            
        print_summary(image_dir=image_dir, ckpt_dir=ckpt_dir)
        

