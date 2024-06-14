import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy

from generator import Generator

def get_random(count):
    return torch.randn(count)

G = Generator()
G.load_state_dict(torch.load('gan.pth'))

f, axarr = plt.subplots(2,3, figsize = (16,8)) # plotting 2 rows by 3 columns; grid of arrays
for i in range(2):
    for j in range(3):
        output = G.forward(get_random(300))
        img = output.detach().numpy() # Detaches tensors to make a numpy array
        img = img.reshape(128,128,3)
        axarr[i,j].imshow(img, interpolation='none', cmap='Greys')

plt.show()