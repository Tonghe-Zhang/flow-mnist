import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

BASE_DIR = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime
def print_summary(image_dir, ckpt_dir):
    # Create the summary message
    summary_message = f"Image saved to: {image_dir}\nCheckpoints saved to: {ckpt_dir}"
    
    # Create a border
    border = "#" * (len(summary_message) + 4)  # 4 for the spaces and borders

    # Print the summary information
    print(border)
    print(f"# {summary_message} #")
    print(border)

def current_time():
    # Get current time
    now = datetime.now()
    # Format the time to the desired pattern
    formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
    return formatted_time

def format_time_seconds(estimated_remaining_time):
    # Assuming estimated_remaining_time is in seconds (float)
    
    # Calculate hours, minutes, and seconds
    hours = int(estimated_remaining_time // 3600)
    minutes = int((estimated_remaining_time % 3600) // 60)
    seconds = int(estimated_remaining_time % 60)
    
    # Format the time string
    formatted_time = f"{hours:02}h:{minutes:02}m:{seconds:02}s"
    
    return formatted_time

def visualize(dataloader, num_samples=25):
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    print(f"Images shape: {images.shape}")
    print(f"Images type: {images.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels type: {labels.dtype}")

    # Convert images to numpy for display
    # images = (images.cpu().numpy()/2.0+0.5)*255.0
    
    # Create a grid of subplots
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(9, 9))
    fig.suptitle("MNIST Samples", fontsize=16)

    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Display image
        ax.imshow(((images[i].cpu()/2+0.5)*255).reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Label: {labels[i].item()}', fontsize=16)

    # Remove any unused subplots
    for i in range(num_samples, rows*cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row][col] if rows > 1 else axes[col])
    plt.tight_layout()
    test_dir =os.path.join(BASE_DIR,'test')
    file_name = "test_mnist.png"
    fig_path = os.path.join(test_dir,file_name)
    plt.savefig(fig_path)
    print(f"file saved to {fig_path}")
    plt.show()

def generate_and_visualize_samples(generator, num_steps:int, ckpt_dir):
    # Iterate through checkpoint files
    from tqdm import tqdm as tqdm
    print(f"testing {os.listdir(ckpt_dir)}")
    image_dir =os.path.join(ckpt_dir, 'generation')
    os.makedirs(image_dir, exist_ok=True)
    for filename in tqdm(os.listdir(ckpt_dir)):
        if filename.endswith('.pth'):
            print(f"testing mode: {filename}")
            ckpt_path = os.path.join(ckpt_dir, filename)
            model_state_dict = torch.load(ckpt_path, weights_only=True)

            # Load the model
            generator.model.load_state_dict(model_state_dict)

            # Generate samples for each class
            fig, axes = plt.subplots(1, 10, figsize=(20, 4))
            fig.suptitle(f"Model Epoch: {filename.split('_')[1].split('.')[0]}", fontsize=16)

            for i, cls in enumerate(range(10)):
                cls = torch.tensor(cls).unsqueeze(0).to(generator.device)
                x_hat = generator.sample(cls, num_steps=num_steps, record_intermediate=False)
                axes[i].imshow((x_hat[0,0].cpu()/2.0+0.5)*225.0, cmap='gray')
                # axes[i].imshow((x_hat[0,0].cpu()*0.5+0.5), cmap='gray')
                axes[i].set_title(f"Class {cls[0].item()}", fontsize=12)
                axes[i].axis('off')
            
            # Save the plot
            file_path = os.path.join(image_dir,f'''{filename.split(".")[0]}.png''')
            plt.savefig(file_path)
            plt.close(fig)
            print(f'''plot saved to {file_path}''')

import torch
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

def calculate_fid(real_samples, generated_samples, device):
    # Load the pre-trained Inception v3 model
    inception_model = models.inception_v3(Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()

    # Transform to have appropriate input size and normalize
    transform = transforms.Compose([
        transforms.Resize(299),  # Inception v3 input size
        transforms.CenterCrop(299),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    def get_features(images):
        images = images.clone().detach()
        # Apply the transformations
        images = transform(images)

        # Send to the specified device
        images = images.to(device)

        # Add batch dimension if a single image, otherwise process as is
        if len(images.shape) == 3:  # If a single image (C, H, W), expand dimensions
            images = images.unsqueeze(0)

        with torch.no_grad():
            # Forward the images through the Inception model and get features
            features = inception_model(images)

            return features

    # Move real and generated samples to the specified device
    real_samples = real_samples.to(device)
    generated_samples = generated_samples.to(device)

    # Extract features
    real_features = get_features(real_samples)
    generated_features = get_features(generated_samples)

    # Calculate mean and covariance of the features
    mu_real = torch.mean(real_features, dim=0).cpu().numpy()
    mu_gen = torch.mean(generated_features, dim=0).cpu().numpy()

    sigma_real = np.cov(real_features.cpu().numpy(), rowvar=False)
    sigma_gen = np.cov(generated_features.cpu().numpy(), rowvar=False)

    # Calculate FID
    mu_diff = mu_real - mu_gen
    cov_sqrt = sqrtm(sigma_real.dot(sigma_gen))
    
    # Handle possible numerical errors when calculating FID
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = mu_diff.dot(mu_diff) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    
    return fid

def plot_generation_process(generator, num_steps:int, cls:int, ckpt_path):
    
    supertitle = ckpt_path.split('/')[-1].split('.')[0]
    
    generator.model.load_state_dict(torch.load(ckpt_path,weights_only=True))

    fig_dir = os.path.join(os.path.dirname(ckpt_path),'generation')
    
    print(f"supertitle={supertitle}, fig_dir={fig_dir}")
    os.makedirs(fig_dir,exist_ok=True)
    
    cls = torch.tensor(cls).unsqueeze(0).to(generator.device)
    x_hat_list = generator.sample(cls, num_steps=num_steps, record_intermediate=True)
    generator
    num_images = len(x_hat_list)
    rows = 8
    cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    fig.suptitle(supertitle, fontsize=16)
    for i, x_hat in enumerate(x_hat_list):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(x_hat[0].cpu() * 225.0, cmap='gray')
        axes[row, col].set_title(f"Index: {i}", fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, supertitle+'_process.png')
    plt.savefig(fig_path)
    print(f"file save at {fig_path}")
    
    
import math
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, warmup_start_lr=1e-8, eta_min=0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of epochs for the warmup phase.
            max_epochs (int): Total number of epochs for training.
            min_lr (float): Minimum learning rate. Default: 1e-6.
            warmup_start_lr (float): Learning rate to start the warmup from. Default: 1e-8.
            eta_min (float): Minimum learning rate for cosine annealing. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs)
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]



class SinusoidalTimeEmbedder(nn.Module):
    def __init__(self, embed_dim=128, max_steps=64):
        super(SinusoidalTimeEmbedder, self).__init__()
        self.embed_dim = embed_dim
        self.max_steps = max_steps
    
    def forward(self, timesteps):
        """
            input:  (N, )
            output: (N, embed_dim)
        """
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes=10, embed_dim=128):
        super(LabelEmbedder, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
    def forward(self, labels):
        """
            input:  (N, )
            output: (N, embed_dim)
        """
        return self.embed(labels)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, tc, frequency_embedding_size=256):
        super(TimestepEmbedder, self).__init__()
        self.hidden_size = hidden_size
        self.tc = tc
        self.frequency_embedding_size = frequency_embedding_size
        self.dense1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.tc.kern_init('time_bias')(self.dense1.bias)
        self.tc.kern_init('time_bias')(self.dense2.bias)

    def forward(self, t):
        x = self.timestep_embedding(t)
        x = self.dense1(x)
        x = F.silu(x)
        x = self.dense2(x)
        return x

    def timestep_embedding(self, t, max_period=10000):
        t = t.float()
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, stop=half, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = embedding.to(self.tc.dtype)
        return embedding
    

class MNISTEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(MNISTEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Flatten layer
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, x):
        """
        input: [N, H, C, W]
        output: [N, latent_dim]
        """
        
        # print(f"Input shape: {x.shape}")  # # print input shape
        
        x = self.encoder[0](x)  # First Conv layer
        # print(f"After Conv1: {x.shape}")  
        
        x = self.encoder[1](x)  # ReLU
        # print(f"After ReLU1: {x.shape}")  
        
        x = self.encoder[2](x)  # MaxPool
        # print(f"After MaxPool1: {x.shape}")  
        
        x = self.encoder[3](x)  # Second Conv layer
        # print(f"After Conv2: {x.shape}")  
        
        x = self.encoder[4](x)  # ReLU
        # print(f"After ReLU2: {x.shape}")  
        
        x = self.encoder[5](x)  # Second MaxPool
        # print(f"After MaxPool2: {x.shape}")  
        
        x = self.encoder[6](x)  # Third Conv layer
        # print(f"After Conv3: {x.shape}")  
        
        x = self.encoder[7](x)  # ReLU
        # print(f"After ReLU3: {x.shape}")  
        
        x = self.encoder[8](x)  # Third MaxPool
        # print(f"After MaxPool3: {x.shape}")  
        
        x = self.encoder[9](x)  # Flatten
        # print(f"After Flatten: {x.shape}")  
        
        x = self.encoder[10](x)  # First Linear
        # print(f"After Linear1: {x.shape}")  
        
        x = self.encoder[11](x)  # ReLU
        # print(f"After ReLU4: {x.shape}")  
        
        x = self.encoder[12](x)  # Second Linear
        # print(f"After Linear2 (latent dim): {x.shape}")  
        
        
        
        return x