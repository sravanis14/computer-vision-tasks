import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt
import time
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("path", type = str, help = "Path of the video file")
args = parser.parse_args()

vid = cv2.VideoCapture(args.path)
num_of_frames = 500
samples = []
count = 0

while(1):
    ret, frame = vid.read()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame1 = cv2.resize(frame1, (256,256))
    if count < num_of_frames:
        samples.append(frame1)
        count += 1
        continue
    else:
        break 
vid.release()
samples = np.array(samples)
print(samples.shape)
samples = samples/255.

background = np.zeros(shape = (samples.shape[1],samples.shape[2], samples.shape[3]))

gmm = GaussianMixture(n_components= 2, max_iter= 100)

t0 = time.time()
for i in range(samples.shape[1]):
    for j in range(samples.shape[2]):
        for k in range(samples.shape[3]):
            X = samples[:, i, j, k]
            X = X.reshape(X.shape[0], 1)
            
            gmm.fit_predict(X)
            means = gmm.means_
            covars = gmm.covariances_
            weights = gmm.weights_
            idx = np.argmax(weights)
            background[i][j][k] = means[idx]

print("Time elapsed :", time.time() - t0)

background = (background*255).astype('uint8')
cv2.imwrite("Backgroundimage1.jpg", background)

cv2.imshow("image",background)
cv2.waitKey(0)
cv2.destroyAllWindows()
