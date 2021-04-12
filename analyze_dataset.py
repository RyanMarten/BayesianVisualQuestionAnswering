import cv2
import os
import numpy as np
import random
#import cPickle as pickle
import pickle
import warnings
import argparse
import matplotlib.pyplot as plt
dirs = './data'

filename = os.path.join(dirs,'sort-of-clevr.pickle')
with  open(filename, 'rb') as f:
	train_dataset, test_dataset = pickle.load(f)
img, ternary_relations, binary_relations, norelations = test_dataset[1]
img = np.array(img)*255
img = img.astype("uint8")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()