import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch import Tensor
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
from model.unet_shortcut import UNetShortCutModelWrapper

class ShortCutReFlow(nn.Module):
    
    def __init__(self, device, data_shape, model, train_cfg, model_path=None):
        super(ShortCutReFlow, self).__init__()
        
        self.train_cfg= train_cfg
        
        self.device=device
        
        self.data_shape=tuple(data_shape)
        
        self.network: UNetShortCutModelWrapper
        self.network = model.to(self.device)
        
        self.lr = train_cfg.lr
        self.optimizer = torch.optim.AdamW(params=self.network.parameters(), lr=self.lr, weight_decay=1e-6)
            
        self.start_epoch = 1

        if model_path is not None:
            self.load_model(model_path)   # possible change self.start_epoch, self.optimizer, and self.network when loaded from checkpoint        

        self.epoch = self.start_epoch-1   # have not started yet. will become self.start_epoch when self.run() starts. 
        
        self.n_epochs= train_cfg.n_epochs
        
        self.end_epoch = self.start_epoch+self.n_epochs-1
        
        self.eval_interval=train_cfg.eval_interval
        
        self.inference_steps = train_cfg.inference_steps
        
        self.max_denoising_steps = train_cfg.max_denoising_steps       
        
        self.use_bc_loss = train_cfg.use_bc_loss
        
        self.batch_size = train_cfg.batch_size
        
        self.use_scheduler = train_cfg.use_scheduler
        if self.use_scheduler:
            self.scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=train_cfg.lr, total_steps=self.n_epochs)
        
        self.alpha = train_cfg.get('alpha', None)                           # bs / flow    order: 100, 10, 1
        self.alpha_scheduler = train_cfg.get('alpha_scheduler', 'constant')
        scheduler_list=['constant', 'balance', 'balance_gradual', 'balance_gradual_sine', 'balance_gradual_innerloop']
        if self.alpha_scheduler not in scheduler_list:
            raise ValueError(f"Invalid alpha_scheduler: {self.alpha_scheduler}. Must be within {scheduler_list}")
        if self.alpha_scheduler == 'constant' and (self.alpha == None or self.alpha < 0):
            raise ValueError("when self.alpha_scheduler == constant, alpha must be specified, and should be non-negative for constant scheduler")
        
        self.eval_fid = True #False
        
        
        self.use_fixed_dt_input = False
        
        
        # self.scheduler = CosineAnnealingLR(optimizer=self.optimizer,
        #                                    T_max=self.n_epochs)
        # self.scheduler = CosineAnnealingWarmupLR(
        #             self.optimizer,
        #             warmup_epochs=train_cfg.warmup_epochs,
        #             max_epochs=train_cfg.max_epochs,
        #             min_lr=train_cfg.min_lr,
        #             warmup_start_lr=train_cfg.warmup_start_lr
        #             )
        
        print(f"ShortCutFlow object contains self.network={self.network}")
    
    def generate_target_flow(self, x1, cls):
        '''
        inputs:
            x1. tensor. real data. torch.Size([N,C,H,W])
            cls. class label. torch.Size([N])
        outputs:
            (xt, t, cls): tuple, the inputs to the model. containing...
                xt: corrupted data. torch.  torch.Size([N,C,H,W])
                t:  corruption ratio        torch.Size([N])
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

        # random generation step label
        max_value = int(np.log2(self.max_denoising_steps))
        dt_base = torch.randint(1, max_value + 1, (self.batch_size,)).to(self.device)
        
        return (xt, t, dt_base), v
    
    
    def generate_bs_target(self, x1:Tensor,cls:Tensor):
        '''
        Generate supervision for the self-distillation of one-step flow via bootstrapping.  
        If you take two steps consecutively, it should be equivalent to taking a larger stride with the magnitude of two steps. 
        
        (i) s(x_t, t, m) * dt +  s(x_t', t+dt, m)* dt  =   s(x_t, t, m/2) * 2dt
            where x_t' = x_t + s(x_t, t, m) * dt, 
        
        or equivalently, 
        
        (ii) 1/2 [ v(x_t, t, log 2m) +  v(x_t', t+dt/2, log 2m) ]   =   v(x_t, t, log m)
        
             where x_t' = x_t + v(x_t, t, 2m) * dt/2, and the LHS is the bootstrap target. 
             
        in (ii), 
            log m is the logarithm of the number of denoising steps when we take a larger stride, and we name it `dt_base`
            dt is the euclidean time interval, 
            x_t is the actions(images), the velocity may also condition on the observation (lables)
            t is the time or noise mixture ratio. 
        
        inputs:
        act: torch.Tensor(batch_size, horizon_steps, action_dim)
        obs: torch.Tensor(batch_size, cond_steps, obs_dim)
        
        outputs:
        act_t_bs:   torch.Tensor(batch_size, horizon_steps, action_dim)
        t_bs:       torch.Tensor(batch_size)
        dt_base_bs: torch.Tensor(batch_size)
        vt_bs:      torch.Tensor(batch_size, horizon_steps, action_dim)
        '''
        
        log2_sections=np.log2(self.max_denoising_steps).astype(np.int32)
        dt_base = (log2_sections - 1 - torch.arange(log2_sections)).repeat_interleave(self.batch_size // log2_sections)
        dt_base = torch.cat([dt_base, torch.zeros(self.batch_size - dt_base.shape[0])]).to(self.device)
        dt_sections = torch.pow(2, dt_base)
        
        # 1) =========== Sample dt for bootstrapping ============
        dt = 1 / (2 ** (dt_base))
        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2
        
        # 2) =========== Generate Bootstrap Targets ============
        # 2.0) sample time       
        batch_size= cls.shape[0]
        t = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        for i in range(batch_size):
            high_value = int(dt_sections[i].item())  # Convert to int since dt_sections contains float values
            t[i] = torch.randint(low=0, high=high_value, size=(1,), dtype=torch.float32).item()
        
        t /= dt_sections
        # print(t_bs)
        # print(t_bs.shape)

        t_broadcasted=t.view(x1.shape[0], *[1 for _ in range(len(x1.shape)-1)]) #[self.batch_size, 1, 1, 1...]
        
        # 2.1) create the noise-corrupted action, act_t
        x0 = torch.randn(x1.shape, device=self.device)
        
        
        # print(f"[generate_bs_target] t_broadcasted.shape={t_broadcasted.shape}, x0.shape={x0.shape}, x1.shape={x1.shape}")
        xt = (1 - (1 - 1e-5) * t_broadcasted) * x0 + t_broadcasted * x1
        
        # 2.2) do integration twice to bootstrap the result of integrating once, to get v_bs.
        # print(f"self.device={self.device}")
        # print(f"t_bs.device={t_bs.device}")
        # print(f"act_bs.device={act_t_bs.device}")
        # print(f"dt_base_bootstrap.device={dt_base_bootstrap.device}")
        # print(f"obs_bs.device={obs_bs.device}")
        
        # print(f"[generate_bs_target] xt.shape={xt.shape}, t.shape={t.shape}, dt_base_bootstrap.shape={dt_base_bootstrap.shape}, cls.shape={cls.shape}")
        v1 = self.network(xt, t, dt_base_bootstrap, cls)
        t2 = t + dt_bootstrap
        
        x2 = xt + dt_bootstrap.view(batch_size,1,1,1)* v1   # image: [N, C, H, W]
        
        v_b2 = self.network(x2, t2, dt_base_bootstrap, cls)
        v_bs = (v1 + v_b2) / 2       
        
        # print(f"M={self.max_denoising_steps}")
        # print(f"dt_base_bs={dt_base_bs}")
        # print(f"dt_sections={dt_sections}")

        return (xt, t, dt_base), v_bs

    
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
        
        x_hat=torch.randn((sample_batch_size,)+self.data_shape, device=self.device)    # [batch_size, C, H, W]
        
        dt = (1/inference_steps)* torch.ones_like(x_hat).to(self.device)
        
        steps = torch.linspace(0,1,inference_steps).repeat(sample_batch_size, 1).to(self.device)                       # [batch_size, num_steps]
        
        dt_base = torch.tensor(np.log2(inference_steps)).repeat(sample_batch_size).to(self.device)
        for i in range(inference_steps):
            t = steps[:,i]
            vt=self.network(x_hat,t,dt_base, cls)
            x_hat+= vt* dt            
            if record_intermediate:
                x_hat_list[i] = x_hat
        return x_hat_list if record_intermediate else x_hat

    def loss_flow(self, xt, t, dt_base, cls, v):
        v_hat = self.network(xt, t, dt_base, cls)
        
        loss_flow = F.mse_loss(input=v_hat, target=v)
        
        return loss_flow
    
    def loss_bs(self, xt_bs, t_bs, dt_base, cls, v_bs):
        v_hat = self.network(xt_bs, t_bs, dt_base, cls) 
        
        loss_bs = F.mse_loss(input=v_hat, target=v_bs)
        
        return loss_bs

    def loss_scheduler(self,loss_flow, loss_bs, epoch, step):
        
        if self.alpha_scheduler == 'constant':
            loss = loss_flow + self.alpha * loss_bs

        elif self.alpha_scheduler == 'balance':
            loss_flow_weight = loss_flow.item()/(loss_flow.item() + loss_bs.item())
            loss_bs_weight = loss_bs.item()/(loss_flow.item() + loss_bs.item())
            loss = loss_flow_weight * loss_flow  + loss_bs_weight * loss_bs

        elif self.alpha_scheduler == 'balance_gradual': # through out different epochs, the bs loss gradualy becomes 1/n_epochs, 2/n_epochs, ...1.0 of the flow loss. 
            progress_percentage = epoch / self.n_epochs
            loss_flow_weight = loss_flow.item()/(loss_flow.item() + loss_bs.item())
            loss_bs_weight = loss_bs.item()/(loss_flow.item() + loss_bs.item())
            loss = loss_flow_weight * loss_flow  + loss_bs_weight * loss_bs * progress_percentage

        
        elif self.alpha_scheduler == 'balance_gradual_sine':
            progress_percentage = epoch / self.n_epochs
            loss_flow_weight = loss_flow.item()/(loss_flow.item() + loss_bs.item())
            loss_bs_weight = loss_bs.item()/(loss_flow.item() + loss_bs.item())
            loss = loss_flow_weight * loss_flow  + loss_bs_weight * loss_bs * np.sin(np.pi/2*progress_percentage)
        
        
        elif self.alpha_scheduler == 'balance_gradual_innerloop': # in the same epoch, the bs loss gradualy becomes 1/n_epochs, 2/n_epochs, ...1.0 of the flow loss. Then repeat for next epoch. 
            progress_percentage = step / self.num_steps_per_epoch
            loss_flow_weight = loss_flow.item()/(loss_flow.item() + loss_bs.item())
            loss_bs_weight = loss_bs.item()/(loss_flow.item() + loss_bs.item())
            loss = loss_flow_weight * loss_flow  + loss_bs_weight * loss_bs * progress_percentage
        return loss
    
    
    def run(self, cfg, train_loader: DataLoader, test_loader: DataLoader):
        self.prepare_run(train_loader, cfg)

        for epoch in tqdm(range(self.start_epoch, self.end_epoch+ 1, 1)):
            self.epoch = epoch
            
            start_time = time.time()  # Start time for the epoch
            
            self.network.train()
            
            self.prepare_train_log()
            
            if self.use_bc_loss:
                self.epoch_bc_loss = []
            
            with tqdm(total=len(train_loader), desc=f'Train epoch {self.epoch}/step: ') as pbar:
                for step, (x, cls) in enumerate(train_loader):
                    x.to(self.device)
                    cls.to(self.device)
                    self.optimizer.zero_grad()
                    
                    (xt, t, dt_base_flow), v = self.generate_target_flow(x, cls)
                    loss_flow = self.loss_flow(xt, t, dt_base_flow, cls, v)
                    
                    (xt_bs, t_bs, dt_base_bs), v_bs = self.generate_bs_target(x, cls)
                    loss_bs = self.loss_bs(xt_bs, t_bs, dt_base_bs, cls, v_bs)
                    
                    loss = self.loss_scheduler(loss_flow, loss_bs, epoch=self.epoch, step=step)
                    
                    if self.use_bc_loss:
                        bc_loss = F.mse_loss(xt, x)
                        loss+=bc_loss
                    
                    loss.backward()
                    
                    self.optimizer.step()
                    
                    self.record_train_loss(locals())

                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'loss_flow': f'{loss_flow:.4f}', 'loss_bs': f'{loss_bs:.4f}'})
            
            self.calculate_epoch_statistics()

            if self.epoch % self.eval_interval == 0:
                # evaluate model
                self.network.eval()
                
                self.prepare_eval_log()
                
                with torch.no_grad():
                    
                    self.generate_and_visualize(self.epoch, file_path=os.path.join(self.ckpt_dir, f'model_{self.epoch}.png'))
                    
                    for i, (x, cls) in tqdm(enumerate(test_loader)):
                            x.to(self.device)
                            cls.to(self.device)
                            x_hat = self.sample(cls=cls, inference_steps=self.inference_steps, record_intermediate=False)

                            mse_err = F.mse_loss(x_hat, x)
                            self.err_list.append(mse_err.item())

                            if self.eval_fid:
                                if i < 3: # to save time. 
                                    fid = calculate_fid(x_hat.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1), device=self.device)
                                    self.fid_list.append(fid)

                            (xt, t, dt_base_flow), v = self.generate_target_flow(x, cls)
                            loss_flow = self.loss_flow(xt, t, dt_base_flow, cls, v)
                        
                            (xt_bs, t_bs, dt_base_bs), v_bs = self.generate_bs_target(x, cls)
                            loss_bs = self.loss_bs(xt_bs, t_bs, dt_base_bs, cls, v_bs)
                            
                            loss_eval = loss_flow + loss_bs
                            
                            self.record_eval_loss(loss_eval, loss_flow, loss_bs) 

                    
                self.plot_statistics(start_time)
                
                self.save_eval_data(locals())
                
                self.save_model()
                
            # if self.epoch % self.update_ema_freq == 0:
            #     self.step_ema()
            
            if self.use_scheduler:
                self.scheduler.step()  # Update the learning rate scheduler
                self.lr = self.scheduler.get_last_lr()[0]
            else:
                self.lr = self.optimizer.param_groups[0]['lr'] # or: for multiple parameter groups. [param_group['lr'] for param_group in optimizer.param_groups]
            
        print_summary(image_dir=self.image_dir, ckpt_dir=self.ckpt_dir)
        
        self.generate_and_visualize_intermediate(fig_path=os.path.join(self.image_dir,'process.png'))
    
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
        print(f'''plot saved to {file_path}\n''')
    
    def generate_and_visualize_intermediate(self, fig_path):
        print(f"generate_and_visualize_intermediate steps for {self.__class__.__name__}")
        from tqdm import tqdm
        # Generate samples for each class with intermediate stages
        if self.inference_steps<5:
            intermediate_steps = list(range(self.inference_steps))
        else:
            intermediate_steps = [0, int(self.inference_steps * 0.2), int(self.inference_steps * 0.4), int(self.inference_steps * 0.6), int(self.inference_steps * 0.8), self.inference_steps - 1]
        num_intermediate = len(intermediate_steps)

        # Create a figure with 5 rows and 10 columns
        fig, axes = plt.subplots(num_intermediate, 10, figsize=(20, 10))
        fig.suptitle(f"ShortCutFlow Generation", fontsize=16)

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
        print(f'Plot saved to {fig_path}\n')
    
    def plot_statistics(self, start_time):
        if len(self.eval_loss_list) <=0:
            raise ValueError("eval_loss_list must be a list")
        self.eval_loss_mean = np.mean(self.eval_loss_list) 
        self.eval_loss_std = np.std(self.eval_loss_list)
        self.eval_losses.append((self.eval_loss_mean, self.eval_loss_std))
        
        eval_loss_f_mean = np.mean(self.eval_loss_f_list) 
        eval_loss_f_std = np.std(self.eval_loss_f_list)
        self.eval_losses_f.append((eval_loss_f_mean, eval_loss_f_std))
        
        
        eval_loss_bs_mean = np.mean(self.eval_loss_bs_list) 
        eval_loss_bs_std = np.std(self.eval_loss_bs_list)
        self.eval_losses_bs.append((eval_loss_bs_mean, eval_loss_bs_std))
        

        self.reconstruct_loss_mean = np.mean(self.err_list)
        self.reconstruct_loss_std = np.std(self.err_list)
        self.reconstruct_losses.append((self.reconstruct_loss_mean, self.reconstruct_loss_std))

        fid_mean = np.mean(self.fid_list)
        fid_std = np.std(self.fid_list)
        self.fid_scores.append((fid_mean, fid_std))

        eval_epochs = np.arange(start=self.eval_interval, stop=self.epoch + 1, step=self.eval_interval)

        # Calculate stats for logging
        epoch_duration = time.time() - start_time  # Duration of the epoch
        remaining_epochs = self.end_epoch - self.epoch
        estimated_remaining_time = remaining_epochs * epoch_duration
        
        # Print logs in mean ± std format
 
        print(f'Epoch [{self.epoch}/{self.end_epoch}] \n'+ \
            f'Train Loss: {self.train_loss_mean:.4f} ± {self.train_loss_std:.4f}\n ' + \
            f'Train Loss_f: {self.train_loss_f_mean:.4f} ± {self.train_loss_f_std:.4f}\n ' + \
            f'Train Loss_b: {self.train_loss_b_mean:.4f} ± {self.train_loss_b_std:.4f}\n ' + \
            f'Train loss BC: {self.bc_loss_mean:.4f} ± {self.bc_loss_std:.4f}\n ' if self.use_bc_loss else f''+ \
            f'Eval Loss: {self.eval_loss_mean:.4f} ± {self.eval_loss_std:.4f}\n ' + \
            f'Reconstruction: {self.reconstruct_loss_mean:.4f} ± {self.reconstruct_loss_std:.4f}\n ' + \
            f'FID score: {fid_mean:.4f} ± {fid_std:.4f}\n '  if self.eval_fid else f'' + \
            f'Learning rate: {self.lr:4f}\n ' + \
            f'Time: {epoch_duration:.2f}s\n ' + \
            'Estimated Remaining Time:' + format_time_seconds(estimated_remaining_time))
        
        # Plot the loss, eval curves with shaded variances
        train_means, train_stds = zip(*self.train_losses)
        train_means_f, train_stds_f = zip(*self.train_losses_f)
        train_means_b, train_stds_b = zip(*self.train_losses_b)
        
        if self.use_bc_loss:
            bc_means, bc_stds = zip(*self.bc_losses)
        
        eval_means, eval_stds = zip(*self.eval_losses)
        eval_means_f, eval_stds_f = zip(*self.eval_losses_f)
        eval_means_bs, eval_stds_bs = zip(*self.eval_losses_bs)
        
        reconstruct_means, reconstruct_stds = zip(*self.reconstruct_losses)
        fid_means, fid_stds = zip(*self.fid_scores)
        
        plt.figure(figsize=(10, 10))

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Subplot 1
        axs[0, 0].plot(self.train_epochs, train_means, label='Train Loss', color='black')
        axs[0, 0].fill_between(self.train_epochs,
                            np.array(train_means) - np.array(train_stds),
                            np.array(train_means) + np.array(train_stds),
                            color='black', alpha=0.2)

        axs[0, 0].plot(eval_epochs, eval_means, label='Eval Loss', color='red')
        axs[0, 0].fill_between(eval_epochs,
                            np.array(eval_means) - np.array(eval_stds),
                            np.array(eval_means) + np.array(eval_stds),
                            color='red', alpha=0.2)
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        # Subplot 2
        # axs[0, 1].plot(eval_epochs, reconstruct_means, label='Reconstruct Error', color='blue')
        # axs[0, 1].fill_between(eval_epochs,
        #                     np.array(reconstruct_means) - np.array(reconstruct_stds),
        #                     np.array(reconstruct_means) + np.array(reconstruct_stds),
        #                     color='blue', alpha=0.2)

        # if self.use_bc_loss:
        #     axs[0, 1].plot(self.train_epochs, bc_means, label='Train BC Loss', color='orange')
        #     axs[0, 1].fill_between(self.train_epochs,
        #                         np.array(bc_means) - np.array(bc_stds),
        #                         np.array(bc_means) + np.array(bc_stds),
        #                         color='orange', alpha=0.2)
        if self.eval_fid:
            axs[0, 1].plot(eval_epochs, fid_means, label='FID', color='purple')
            axs[0, 1].fill_between(eval_epochs,
                                np.array(fid_means) - np.array(fid_stds),
                                np.array(fid_means) + np.array(fid_stds),
                                color='purple', alpha=0.2)
        axs[0, 1].set_title('FID score')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        
        # Subplot 3
        axs[1, 0].plot(self.train_epochs, train_means_f, label='Train Loss f', color='black')
        axs[1, 0].fill_between(self.train_epochs,
                            np.array(train_means_f) - np.array(train_stds_f),
                            np.array(train_means_f) + np.array(train_stds_f),
                            color='black', alpha=0.2)

        axs[1, 0].plot(eval_epochs, eval_means_f, label='Eval Loss f', color='red')
        axs[1, 0].fill_between(eval_epochs,
                                np.array(eval_means_f) - np.array(eval_stds_f),
                                np.array(eval_means_f) + np.array(eval_stds_f),
                                color='red', alpha=0.2)
        axs[1, 0].set_title('Loss f')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

        # Subplot 4
        axs[1, 1].plot(self.train_epochs, train_means_b, label='Train Loss b', color='black')
        axs[1, 1].fill_between(self.train_epochs,
                            np.array(train_means_b) - np.array(train_stds_b),
                            np.array(train_means_b) + np.array(train_stds_b),
                            color='black', alpha=0.2)

        axs[1, 1].plot(eval_epochs, eval_means_bs, label='Eval Loss bs', color='red')
        axs[1, 1].fill_between(eval_epochs,
                            np.array(eval_means_bs) - np.array(eval_stds_bs),
                            np.array(eval_means_bs) + np.array(eval_stds_bs),
                            color='red', alpha=0.2)
        axs[1, 1].set_title('Loss b')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].legend()

        plt.suptitle(f"Training {self.__class__.__name__} with alpha={self.alpha_scheduler if self.alpha_scheduler != 'constant' else str(self.alpha)}")
        plt.tight_layout()

        plt.savefig(os.path.join(self.image_dir, 'loss.png'))
        print(f"loss curves saved to {os.path.join(self.image_dir, 'loss.png')}")
        plt.close()  # Close the figure to free up memory
    
    
    def prepare_run(self, train_loader, cfg):
        self.num_steps_per_epoch = len(train_loader)
        self.train_losses = []
        self.train_losses_f = []
        self.train_losses_b = []
        self.train_loss_f_mean=0.0
        if self.use_bc_loss:
            self.bc_losses = []
        self.eval_losses = []
        self.eval_losses_f = []
        self.eval_losses_bs= []
        self.reconstruct_losses = []
        self.fid_scores = []

        self.log_dir = os.path.join(BASE_DIR, 'log', self.__class__.__name__, current_time()+'_'+self.alpha_scheduler+'_'+str(self.alpha))
        self.image_dir = os.path.join(self.log_dir, 'visualize')
        self.ckpt_dir = os.path.join(self.log_dir, 'ckpt')
        self.cfg_path = os.path.join(self.ckpt_dir, 'cfg.yaml')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # dump the config file used to train. 
        OmegaConf.save(cfg, self.cfg_path)
    
    


    def prepare_train_log(self):
        self.epoch_train_loss = []
        self.epoch_train_loss_f = []
        self.epoch_train_loss_b = []
    def record_train_loss(self, locals):
        loss=locals.get('loss')
        loss_flow=locals.get('loss_flow')
        loss_bs=locals.get('loss_bs')
        
        self.epoch_train_loss.append(loss.item())
        self.epoch_train_loss_f.append(loss_flow.item())
        self.epoch_train_loss_b.append(loss_bs.item())
        
        if self.use_bc_loss:
            bc_loss=locals.get('bc_loss')
            self.epoch_bc_loss.append(bc_loss.item())
        # print(f"epoch: {epoch}, steps: {step}, loss: {loss.item():.4}", end="")
    
    def calculate_epoch_statistics(self):
        # Calculate average training loss for this epoch
        self.train_loss_mean = np.mean(self.epoch_train_loss)
        self.train_loss_std = np.std(self.epoch_train_loss)
        self.train_losses.append((self.train_loss_mean, self.train_loss_std))
        
        self.train_loss_f_mean = np.mean(self.epoch_train_loss_f)
        self.train_loss_f_std = np.std(self.epoch_train_loss_f)
        self.train_losses_f.append((self.train_loss_f_mean, self.train_loss_f_std))
        
        self.train_loss_b_mean = np.mean(self.epoch_train_loss_b)
        self.train_loss_b_std = np.std(self.epoch_train_loss_b)
        self.train_losses_b.append((self.train_loss_b_mean, self.train_loss_b_std))
        
        self.train_epochs = np.arange(start=self.start_epoch, stop=self.epoch + 1, step=1)
        
        if self.use_bc_loss:
            self.bc_loss_mean = np.mean(self.epoch_bc_loss)
            self.bc_loss_std = np.std(self.epoch_bc_loss)
            self.bc_losses.append((self.bc_loss_mean, self.bc_loss_std))
    
    def prepare_eval_log(self):
        self.err_list = []
        self.eval_loss_list = []
        self.eval_loss_f_list=[]
        self.eval_loss_bs_list=[]
        self.fid_list = []
    
    def record_eval_loss(self, loss_eval, loss_flow, loss_bs):
        self.eval_loss_list.append(loss_eval.item())
        self.eval_loss_f_list.append(loss_flow.item())
        self.eval_loss_bs_list.append(loss_bs.item())
    
    
    def save_eval_data(self, locals):
        # Save data during evaluation
        data_to_save = {
            'train_epochs': locals.get('train_epochs'),
            'train_means': locals.get('train_means'),
            'train_stds': locals.get('train_stds'),
            'eval_epochs': locals.get('eval_epochs'),
            'eval_means': locals.get('eval_means'),
            'eval_stds': locals.get('eval_stds'),
            'train_means_f': locals.get('train_means_f'),
            'train_stds_f': locals.get('train_stds_f'),
            'eval_means_f': locals.get('eval_means_f'),
            'eval_stds_f': locals.get('eval_stds_f'),
            'train_means_b': locals.get('train_means_b'),
            'train_stds_b': locals.get('train_stds_b'),
            'eval_means_bs': locals.get('eval_means_bs'),
            'eval_stds_bs': locals.get('eval_stds_bs'),
            'reconstruct_means': locals.get('reconstruct_means'),
            'reconstruct_stds': locals.get('reconstruct_stds')
        }

        if self.use_bc_loss:
            data_to_save.update({
            'bc_means': locals.get('bc_means'),
            'bc_stds': locals.get('bc_stds')
            })

        if self.eval_fid:
            data_to_save.update({
            'fid_means': locals.get('fid_means'),
            'fid_stds': locals.get('fid_stds')
            })

        np.savez(os.path.join(self.image_dir,'eval_data.npz'), **data_to_save)
    
    def save_model(self):
        checkpoint = {
            'model_state_dict': self.network.state_dict(),  # Save model weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Save optimizer state
            'epoch': self.epoch
            }
        torch.save(checkpoint, os.path.join(self.ckpt_dir, f'model_{self.epoch}.pth'))
    
    def load_model(self,model_path:str):
        print(f"loading model from {model_path}")
        checkpoint = torch.load(model_path,  weights_only=True)
        
        if 'epoch' in checkpoint.keys():
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"Warning: epoch is not in checkpoint.keys(). Resume from epoch=0")
        if 'model_state_dict' in checkpoint.keys():
            self.network.load_state_dict(checkpoint['model_state_dict'])
            print(f"...successfully loaded model_state_dict to self.network")
        else:
            raise ValueError(f"'model_state_dict' not in checkpoint.keys()={checkpoint.keys()}!")
        if 'optimizer_state_dict' in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"...successfully loaded optimizer_state_dict to self.optimizer")
        else:
            print(f"Warning: optimizer.state_dict not in checkpoint.keys(). Resume optimizer from starting state. s")
        self.network.to(device=self.device)
        print(f"...loading complete.")
    