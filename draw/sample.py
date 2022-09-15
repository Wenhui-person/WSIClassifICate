import numpy as np
import matplotlib.pyplot as plt

img = np.load(r'/home/qianslab/yangwenhui/Multiple Instance Learning manuscript1/code/wsiClassification/draw/probs_map.npy')
img = img.T
plt.imshow(img,cmap=plt.get_cmap('binary_r'))
#plt.colorbar()
plt.axis("off")
plt.savefig('test.png',dpi=650,bbox_inches='tight')
plt.show()