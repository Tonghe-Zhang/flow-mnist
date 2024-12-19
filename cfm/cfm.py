import os

import matplotlib.pyplot as plt
import torch
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from conditional_flow_matching import *
from unet import UNetModel, UNetModelWrapper
def main():
        
    savedir = "models/cond_mnist"
    os.makedirs(savedir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 128
    n_epochs = 10

    trainset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    sigma = 0.0
    # model = UNetModel(
    #     dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True
    # ).to(device)

    model = UNetModelWrapper(
        dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True
    ).to(device)



    optimizer = torch.optim.Adam(model.parameters())
    FM = ConditionalFlowMatcher(sigma=sigma)
    # Users can try target FM by changing the above line by
    # FM = TargetConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x1 = data[0].to(device)
            y = data[1].to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            print(f"t:{t.shape}, x:{xt.shape}, y:{y.shape}")
            '''
            torch.Size([128]), x:torch.Size([128, 1, 28, 28]), y:torch.Size([128])
            '''
            vt = model(t, xt, y)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")
        
        USE_TORCH_DIFFEQ = True
        generated_class_list = torch.arange(10, device=device).repeat(10)
        with torch.no_grad():
            if USE_TORCH_DIFFEQ:
                traj = torchdiffeq.odeint(
                    lambda t, x: model.forward(t, x, generated_class_list),
                    torch.randn(100, 1, 28, 28, device=device),
                    torch.linspace(0, 1, 2, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )
            else:
                traj = node.trajectory(
                    torch.randn(100, 1, 28, 28, device=device),
                    t_span=torch.linspace(0, 1, 2, device=device),
                )
        grid = make_grid(
            traj[-1, :100].view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
        )
        img = ToPILImage()(grid)
        img.save(f'output_image{epoch}.png')

    # plt.imshow(img)
    # plt.show()



if __name__=="__main__":
    main()