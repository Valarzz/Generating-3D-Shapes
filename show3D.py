import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os


def showVoxel(voxel):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxel, edgecolor="k")
    plt.show()


path = "output/output_27600_lossD_-0.05587559938430786_lossG_-0.47137290239334106" + ".npy"
airplane = np.load(path)
print(airplane.shape)
showVoxel(airplane)
