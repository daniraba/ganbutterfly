import os 
import random

dataset = []

dir = 'test'
for sub_dir in os.listdir(dir): #subdirectoies
    files = os.listdir(dir + '/' + sub_dir) #getting the paths
    for file in files: # opening the files
        dataset.append(dir + '/' + sub_dir + '/' + file)

random.shuffle(dataset) # shuffling them

print(dataset)
for i in range(1000):
    image = Image.open(dataset[i])
    print(file)