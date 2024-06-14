import torch
import torch.nn as nn
import numpy as np
import os
import random
from PIL import Image

from discriminator import Discriminator
from generator import Generator

def get_random(count):
    return torch.randn(count)

D = Discriminator()
G = Generator()

dataset = []

dir = 'test'
for sub_dir in os.listdir(dir): #subdirectoies
    files = os.listdir(dir + '/' + sub_dir) #getting the paths
    for file in files: # opening the files
        dataset.append(dir + '/' + sub_dir + '/' + file)

random.shuffle(dataset) # shuffling them
epochs = 1

print(len(dataset))

for epoch in range(epochs):
    for i in range(10000):
        if i % 10 == 0:
            print(i)

        image = Image.open(dataset[i]) # Name of the file and directory
        image = image.resize((128, 128))
        a = np.array(image)/255.0 # Converting to array and grayscale

        a = a.reshape(128*128*3)

        image = torch.FloatTensor(a)
        target = torch.FloatTensor([1.0])

        D.train(image, target)
        g_output = G.forward(get_random(300)).detach()

        target = torch.FloatTensor([0.0])
        D.train(g_output, target)

        target = torch.FloatTensor([1.0])
        G.train(D, get_random(300), target)

torch.save(G.state_dict(), 'gan.pth')