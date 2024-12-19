import os
import sys
from datetime import datetime
from tqdm import tqdm as tqdm

import hydra
from omegaconf import OmegaConf

import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

os.environ['PYTHONPATH']='$PYTHONPATH:/home/zhangtonghe/flow-mnist/data'
# Get the directory of the current script
'''
export PYTHONPATH=$PYTHONPATH:/home/zhangtonghe/flow-mnist/data
export PYTHONPATH=$PYTHONPATH:/home/zhangtonghe/flow-mnist/alg
'''


from alg import ReFlow, ShortCutReFlow
from data.mnist_dataset import *
current_dir = os.path.dirname(os.path.abspath(__file__))
iprt_dirs=[]
iprt_dirs.append(os.path.join(current_dir, '..', 'model'))
iprt_dirs.append(os.path.join(current_dir, '..', 'data'))
for import_dir in iprt_dirs:
    if import_dir not in sys.path:
        sys.path.append(import_dir)


CALCULATE_FID = True

@hydra.main(version_base=None,
            config_path=os.path.join(os.getcwd(),'cfg'),         # can be override with --config-dir=
            config_name='cfg_shortcut.yaml'                            # cfg04.yaml' can be override with --config-name=
            )
def main(cfg:OmegaConf):
    OmegaConf.resolve(cfg)
    model =hydra.utils.instantiate(cfg.algo)
    
    # preparation
    # model: ReFlow # ShortCutReFlow # ReFlow
    test_dir =os.path.join(BASE_DIR,'test',current_time())
    os.makedirs(test_dir,exist_ok=True)
    
    # record the model and its configuration file
    torch.save(model.network.state_dict(), os.path.join(test_dir,'model.pth'))
    OmegaConf.save(cfg, os.path.join(test_dir,'cfg.yaml'))
    
    # generation
    #model.generate_and_visualize_intermediate(fig_path=os.path.join(test_dir,'generate_intermediate.png'))
    
    # test FID
    test_dataset = MNISTDataset(device = cfg.dataset.device, csv_file=os.path.join('data', 'mnist_test.csv'), use_top=cfg.dataset.use_first_eval)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=True)
    num_denoising_steps_list = [1, 2, 4, 8, 16, 20, 32, 64, 128, 256, 512]
    fid_scores = []

    # Enable interactive plotting
    plt.ion()
    plt.figure(figsize=(10, 6))

    fig_dir =os.path.join('eval', model.__class__.__name__, current_time())
    os.makedirs(fig_dir, exist_ok=True)
    # Plot settings
    plt.title(f'{model.__class__.__name__} FID vs. Denoising Steps')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('FID Score')
    plt.xticks(num_denoising_steps_list)
    plt.grid()

    
    pic_list = []
    
    fid_scores = []
    cls_list= []
    # Prepare a real-time plot for FID scores
    plt.ion()  # Enable interactive mode

    x_target = None
    cls_target = None
    for i, (x_, cls_) in enumerate(test_loader):
        if cls_[0].item() == 8:
            x_target=x_
            cls_target = cls_
            
            
    # Iterate over different num_denoising_steps
    with tqdm(total=len(num_denoising_steps_list), postfix={"FID": 0.0}, dynamic_ncols=True) as pbar:
        for num_denoising_steps in tqdm(num_denoising_steps_list):
            fid_list = []
            x = x_target.to(model.device)
            cls = cls_target.to(model.device)
            cls_list.append(cls[0])
            # Sample from the model with the current denoising steps
            x_hat = model.sample(cls=cls, inference_steps=num_denoising_steps, record_intermediate=False)
            fid =calculate_fid(x_hat.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1), device=model.device) if CALCULATE_FID else 0
            fid_list.append(fid)
            pic_list.append(x_hat[0,0]) # Capture the first denoised image

            # Store the mean FID score for current denoising steps and print to progress bar. 
            mean_fid = np.mean(fid_list)
            fid_scores.append(mean_fid)
            pbar.postfix = {"FID": mean_fid}
            pbar.update(1)

            # Update the FID score plot in real time
            plt.figure(1)
            plt.clf()  # Clear the figure to update it
            plt.semilogx(num_denoising_steps_list[:len(fid_scores)], fid_scores, marker='o')
            plt.title('FID vs. Denoising Steps')
            plt.xlabel('Denoising Steps')
            plt.ylabel('FID Score')
            plt.xlim(min(num_denoising_steps_list), max(num_denoising_steps_list))
            plt.ylim(min(fid_scores) - 10, max(fid_scores) + 10)  # Adjust as necessary
            plt.pause(0.1)  # Pause to update the plot

            # Save the plot after each iteration
            plt.savefig(os.path.join(fig_dir, 'FID_vs_denoising_steps.png'))
    
    data_path = os.path.join(fig_dir, 'eval_data.npz')
    np.savez(data_path, fid_scores=np.array(fid_scores), num_denoising_steps_list=np.array(num_denoising_steps_list))
    print(f"\ndata saved to {data_path}. FID scores: {np.array(fid_scores)}\n")
    
    num_images = len(num_denoising_steps_list)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    # Ensure axes is iterable
    if num_images == 1:
        axes = np.array([axes])

    for i in range(num_images):
        # Normalize image appropriately and convert to numpy array for imshow
        image = pic_list[i].cpu().numpy() / 2.0 + 0.5
        axes[i].imshow(image * 255.0, cmap='gray')

        # Setting the title for each image
        axes[i].set_title(f"Step {num_denoising_steps_list[i]}", fontsize=12) # Class {cls_list[i].item()} - 

        # Turn off the axis for each subplot
        axes[i].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure to a file
    fig_path=os.path.join(fig_dir, 'image_denoising_steps.png')
    plt.savefig(fig_path)
    
    print(f"\nfigure saved to {fig_path}\n")
    plt.close(fig)  # Close the figure to free up memory
    
    
    
    '''
    ckpt_path = '/home/zhangtonghe/flow-mnist/log/24-12-07-12-39-37/ckpt/model_100.pth'
    cls= 9 
    plot_generation_process(generator=reflow, 
                            num_steps=cfg.algo.train_cfg.inference_steps, 
                            cls = cls, 
                            ckpt_path =ckpt_path)
    
    
    ckpt_dir = '/home/zhangtonghe/flow-mnist/log/24-12-07-12-39-37/ckpt'
    generate_and_visualize_samples(generator=reflow, 
                                   num_steps=cfg.algo.train_cfg.inference_steps, 
                                   ckpt_dir=ckpt_dir)
    '''
    
if __name__ == '__main__':
    main()