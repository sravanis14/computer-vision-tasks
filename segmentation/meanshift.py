import cv2 
import numpy as np 
from sklearn.cluster import MeanShift, estimate_bandwidth 

path = ".\test_image.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img = cv2.resize(img, (512,512))

X = np.reshape(img, (-1,3))
flat_image = X.astype('float32')
# Estimate the bandwidth
# meanshift
bandwidth = estimate_bandwidth(flat_image, quantile=0.05, n_samples=300)
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True,)
ms.fit(flat_image)
labeled=ms.labels_


# get number of segments
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# get the average color of each segment
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((img.shape))

# show the original
cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# show the result
cv2.imshow('result', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()