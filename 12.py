import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import save_image


train_data = datasets.MNIST("./data", train=True, download=True, 
  transform=T.Compose([T.ToTensor()])
)

dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

z_size = 49

generator = nn.Sequential(
  nn.Linear(z_size, 49 * 3),
  nn.LeakyReLU(0.5),
  nn.BatchNorm1d(49 * 3),
  
  # shape [batch, 49 * 3]
  nn.Reshape((-1, 3, 7, 7)),
  nn.Conv2dTranspose(3, 2, 4, 2, 1),
  nn.LeakyReLU(0.5),
  nn.BatchNorm2d(2),
  
  # shape [batch, 2, 14, 14]
  nn.Conv2dTranspose(2, 1, 4, 2, 1),
  nn.Tanh()
  nn.Reshape((-1, 28, 28)),
  
)


discriminator = nn.Sequential(
  nn.Reshape((-1, 1, 28, 28)),
  nn.Conv2d(1, 2, 4, 2, 1),
  nn.ReLU(),
  nn.BatchNorm2d(2),
  
  # shape [batch, 2, 14, 14]
  nn.Conv2d(2, 3, 4, 2, 1),
  nn.ReLU(),
  nn.BatchNorm2d(2),
  
  # shape [batch, 3, 7, 7]
  nn.Flatten(),
  
  # shape [batch, 3 * 7 * 7]
  nn.Linear(3 * 7 * 7, 1),
  nn.Sigmoid()
)


g_optim = torch.optim.Adam(generator.parameters())
d_optim = torch.optim.Adam(discriminator.parameters())

loss_crit = torch.nn.BSELoss()

losses = []
for i, (images, _) in enumerate(tqdm(dataloader, leave=False)):
  b_size = images.shape[0]
  ones = torch.ones((b_size, 1))
  zeros = torch.zeros((b_size, 1))
  
  # train the generator
  g_optim.zero_grad()
  z = torch.rand((b_size, z_size))
  fake_imgs = generator(z)
  g_loss = loss_crit(discriminator(fake_imgs), ones)
  g_loss.backward()
  g_optim.step()
  
  # train the discriminator
  d_optim.zero_grad()
  d_loss = (
    loss_crit(discriminator(images), ones)
    + 
    loss_crit(discriminator(fake_imgs), zeros)
  )
  d_loss.backward()
  d_optim.step()
  
  losses.append([g_loss.item(), d_loss.item()])
  
  if i % 1000 == 0:
    save_image(gen_imgs.data[:8], f"generated/{i}.png")
  
plt.plot(losses[:, 0])
plt.plot(losses[:, 1])
plt.show()