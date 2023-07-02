import os
import sys
import torch
import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__== "__main__":
    # -- get arguments
    input_dataset = sys.argv[1]
    # -- load dataset
    dataset  = torch.load(input_dataset).detach().numpy()
    # -- visualize
    while True:
        frame = int(input("enter number of frame to be visualized: "))
        # -- plot image
        print(dataset.shape)
        plt.imshow(dataset[frame,:,:,2], cmap=mpl.colormaps["seismic"])
        #plt.imshow(dataset[0,frame,:,:,4], cmap=mpl.colormaps["seismic"])
        plt.axis('off')
        plt.show()

