import numpy as np
import matplotlib.pyplot as plt
from .utilities import compute_lbp_features, minRotateValue
from tqdm import tqdm

class LBP:
    def __init__(self, rotation_invariant = False):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.rotation_invariant = rotation_invariant
        self.minHistoValue = None
        
        if rotation_invariant:
            self.minHistoValue = minRotateValue(256)

    def load_image_lbp_features(self, X, isTrain = True):
        feat = []
        t_bar = tqdm(X)
        t = "Rotataional Invariant |" if self.rotation_invariant else "Standard LBP |"
        if isTrain:
            t_bar.set_description(f"{t} Extracting Train Features")
        else:
            t_bar.set_description(f"{t} Extracting Test Features")

        for img in t_bar:
            img = np.array(img).astype(np.int16)
            hist, lbp_map = compute_lbp_features(img, self.rotation_invariant, self.minHistoValue)
            feat.append(hist)
        return np.array(feat)
    
    def classifier_1nn(self):
        y_pred = []
        for x in self.X_test:
            diff = self.X_train - x
            dist = np.linalg.norm(diff, axis = 1)
            idx = np.argmin(dist)
            y_pred.append(self.y_train[idx])
        return y_pred
    
    def fit(self, X_train, y_train):
        self.X_train = self.load_image_lbp_features(X_train, isTrain = True)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test, y_test):
        self.X_test= self.load_image_lbp_features(X_test, isTrain = False)
        self.y_test = np.array(y_test)
        self.y_pred = self.classifier_1nn()
    
