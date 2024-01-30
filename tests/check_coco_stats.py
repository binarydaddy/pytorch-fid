import numpy as np

coco_512 = np.load('/data2/coco2017/coco2017_val_512_stats.npz')
coco_256 = np.load('/data2/coco2017/coco2017_val_256_stats.npz')


for key in coco_512.keys():
    print(f"{key}, {coco_512[key].shape = }")

for key in coco_256.keys():
    print(f"{key}, {coco_256[key].shape = }")


mu_diff = np.mean(coco_512['mu'] - coco_256['mu'])
sigma_diff = np.mean(coco_512['sigma'] - coco_256['sigma'])