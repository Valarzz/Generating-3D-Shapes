import os
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt


def get_mat(path):
    all = ["train", "test"]
    files = []
    for i in all:
        path2airplane = os.path.join(path, i)
        paths = os.listdir(path2airplane)
        paths = [os.path.join(path2airplane, path) for path in paths]
        files.extend(paths)
    return files


path = "volumetric_data/airplane/30"
files = get_mat(path)

print(len(files))

for id, file in enumerate(files):
    voxels = io.loadmat(file)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    np.save("airplaneData/airplane_"+str(id)+".npy", voxels)

