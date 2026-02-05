import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import random
from PIL import Image
from tqdm import tqdm

def compute_lbp_features(image, rotation_invariant = False, minHistoValue = None):
    centerPixel = image[1:-1, 1:-1]
    neighbor_img = [
        image[:-2, :-2],
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, 2:],
        image[2:, 2:],
        image[2:, 1:-1],
        image[2:, :-2],
        image[1:-1, :-2]
    ]
    res = np.zeros(centerPixel.shape, dtype=np.uint8)
    
    for i in range(8):
        res += ((neighbor_img[i] >= centerPixel).astype(np.uint8)) * (2 ** i)
    
    # mapping feature value to the min rotated value
    if rotation_invariant:
        if minHistoValue is None:
            minHistoValue = minRotateValue(256)
        res = minHistoValue[res]

    hist, bin = np.histogram(res.flatten(), bins = 256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    return hist, res

def load_data(path, classes, rotate_angle = 0, isTrain = True):
    X = []
    y = []
    t_bar = tqdm(classes)

    if isTrain:
        t_bar.set_description("Loading Train Images")
    else:
        t_bar.set_description("Loading Test Images")

    for label in t_bar:
        t_bar.set_postfix({'Current Class': label})
        label_path = os.path.join(path, label)
        images = os.listdir(label_path)
        for image_name in images:
            img = Image.open(os.path.join(label_path, image_name)).convert('L')
            # Rotating the image
            img = img.rotate(rotate_angle, resample = Image.Resampling.BILINEAR)

            X.append(np.array(img).astype(np.int16))
            y.append(label)

    return X,y


def load_train_test_data(train_path, test_path, train_rotate_angle = 0, test_rotate_angle = 0, n_class = None):
    if not os.path.exists(train_path):
            print(f"Error: Train Path not found: {train_path}")
            return np.array([]), np.array([])
    if not os.path.exists(test_path):
            print(f"Error: Test Path not found: {test_path}")
            return np.array([]), np.array([])
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    if n_class is not None:
        classes = sorted(random.sample(os.listdir(train_path), n_class))
    else:
        classes = sorted(os.listdir(train_path))

    X_train, y_train = load_data(train_path, classes, train_rotate_angle, isTrain = True)
    X_test, y_test = load_data(test_path, classes, test_rotate_angle, isTrain = False)

    return X_train, X_test, y_train, y_test
    

def getAccuracy(y_pred, y_test):
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if isinstance(y_test, list):
        y_test = np.array(y_test)

    return np.sum(y_pred == y_test) / y_pred.size

def createConfusionMatrix(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred, labels=sorted(list(set(y_test))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(list(set(y_test))))

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def rotateBitsRight(n, rotations, width):
    mask = (1 << width) - 1
    n &= mask
    rotations %= width
    return ((n >> rotations) | (n << (width - rotations))) & mask

def minRotateValue(n = 256):
    minValueHash = np.arange(0, 256, dtype=np.uint8)
    for i in range(n):
        for r in range(8):
            value = rotateBitsRight(i, r, 8)
            if value < minValueHash[i]:
                minValueHash[i] = value
    return minValueHash
