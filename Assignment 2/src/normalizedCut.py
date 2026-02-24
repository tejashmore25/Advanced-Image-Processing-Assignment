import numpy as np
import matplotlib.pyplot as plt
from skimage import color, transform, filters
from scipy.linalg import eigh
from sklearn.cluster import KMeans

class NormalizedCut:
    def __init__(self, sigmaI=0.01, sigmaX=10, sigmaE=0.5):
        self.sigma_I = sigmaI
        self.sigma_X = sigmaX
        self.sigma_E = sigmaE

        # feature vectors
        self.I = None
        self.X = None
        self.E = None

        # storing the HxW of the resized image
        self.original_shape = None
        
        #Matrix
        self.W = None
        self.D = None

        self.evals = None
        self.evecs = None

    def fit(self, image, resize=(40, 40)):
        img_resized = transform.resize(image, resize, anti_aliasing=True)
        self.original_shape = img_resized.shape[:2]

        if len(img_resized.shape) == 3:
            img_resized = color.rgb2gray(img_resized)

        edges = filters.sobel(img_resized)
        rows, cols = img_resized.shape
        x_cords, y_cords = np.meshgrid(np.arange(cols), np.arange(rows))

        self.I = img_resized.flatten()
        self.X = np.vstack((x_cords.flatten(), y_cords.flatten())).T
        self.E = edges.flatten()

        return self
    
    def build_W(self, use_edge=False):
        diff_I = (self.I[:, np.newaxis] - self.I[np.newaxis, :]) ** 2
        diff_X = np.sum((self.X[:, np.newaxis, :] - self.X[np.newaxis, :, :])**2, axis = 2)
        W = np.exp(-diff_I / (self.sigma_I ** 2)) * np.exp(-diff_X / (self.sigma_X ** 2))

        if use_edge:
            diff_E = (self.E[:, np.newaxis] - self.E[np.newaxis, :]) ** 2
            W *= np.exp(-diff_E / (self.sigma_E ** 2))
        self.W = W
        return W
    
    def solve_nCut(self, K = 5):
        row_sum = np.sum(self.W, axis = 1)
        self.D = np.diag(row_sum)
        L = self.D - self.W

        evals, evecs = eigh(L, self.D)

        idx = np.argsort(evals)
        self.evals = evals[idx]
        self.evecs = evecs[:, idx]
        self.evecs = self.evecs[:, 1:K+1]
        return self.evals, self.evecs

    def applyThreshold(self, evec):
        threshold = np.median(evec)
        mask = evec > threshold
        return mask
    
    def partition_twoWay(self):
        smallest_evec = self.evecs[:, 0]
        segment_mask = self.applyThreshold(smallest_evec).astype(int)
        segment_img = segment_mask.reshape(self.original_shape)
        
        # for consistent background and foreground
        if segment_img[0,0] == 1:
            segment_img = 1 - segment_img
        return segment_img
    
    def partition_recursive_Kway(self, target_K, current_K = 1, mask = None, labels = None):
        if labels is None:
            labels = np.zeros(self.I.shape, dtype = int)
            mask = np.ones(self.I.shape, dtype = bool)

        if current_K >= target_K:
            return labels
        
        idx = np.where(mask)[0]
        # Base case
        if len(idx) < 5:
            return labels
        
        W_sub = self.W[np.ix_(idx, idx)]
        D_sub = np.diag(np.sum(W_sub, axis = 1))
        L_sub = D_sub - W_sub

        evals, evecs = eigh(L_sub, D_sub)
        smallest_evec = evecs[:, 1]
        segment_mask = self.applyThreshold(smallest_evec)

        obj_region = idx[segment_mask]
        labels[obj_region] = current_K

        next_mask = np.zeros_like(mask)
        next_mask[idx[segment_mask == False]] = True

        labels = self.partition_recursive_Kway(target_K, current_K + 1, next_mask, labels)
        if current_K == 1:
            return labels.reshape(self.original_shape)
        return labels
    
    def partition_Ksimultaneous(self, K = 3):
        model_kmeans = KMeans(n_clusters=K, random_state=25, n_init=10)
        features = self.evecs[:, :K-1]
        labels = model_kmeans.fit_predict(features)
        return labels.reshape(self.original_shape)
    
        


        


    