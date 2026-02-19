import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from utils import *
from torch.utils.data import TensorDataset, DataLoader


class EdgeMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim_input = 16, hidden_dim_output = 8, output_dim = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_input),
            nn.ReLU(),
            nn.Linear(hidden_dim_input, hidden_dim_output),
            nn.ReLU(),
            nn.Linear(hidden_dim_output, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def feature_extraction(path):

    sobel_x = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]], np.float32)
    sobel_y = np.array([[1,2,1],
               [0,0,0],
               [-1,-2,-1]], np.float32)
    
    img = load_picture(path)
    deriv_x = conv(img, sobel_x)
    deriv_y = conv(img, sobel_y)
    strength = np.sqrt(deriv_x ** 2 + deriv_y ** 2)
    strength = np.float32(strength / strength.max() * 255)
    strength = strength.flatten()
    strength = strength[..., None]

    out = torch.from_numpy(strength)
    return out

def get_dataloader(basePath):
    basePath = abspath(basePath)

    X_trainPath = join(basePath, "images/train")
    X_testPath = join(basePath, "images/test")
    y_trainPath = join(basePath, "groundTruth/train")
    y_testPath = join(basePath, "groundTruth/test")

    X_trainPaths = get_image_paths(X_trainPath)
    X_testPaths = get_image_paths(X_testPath)

    feats = []
    for i, trainPath in enumerate(X_trainPaths):
        feat = feature_extraction(trainPath)
        feats.append(feat)
        if i > 10:
            break

    conts = []
    contours = load_contours(y_trainPath)
    i = 0
    for _, contour in contours.items():
        conts.append(np.array(contour[0]).flatten())
        if i > 10:
            break
    

    X_train = torch.tensor(np.array(feats).flatten(), dtype=torch.float32)
    y_train = torch.tensor(np.array(conts).flatten(), dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, shuffle=False)

    return train_loader


    return 

if __name__ == "__main__":
    

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    paths = get_image_paths("./cv-parktikum/BSDS500-master/BSDS500/data/images/test")
    feature = feature_extraction(paths[0])
    contours = load_contours("./cv-parktikum/BSDS500-master/BSDS500/data/groundTruth/test/")
    contour = contours[basename(paths[0]).split('.')[0]]
    print(feature.shape)
    mlp = EdgeMLP()
    result = mlp.forward(feature)
    print("Result")
    print(result)
    print("contour")
    print(contour)
    loader = get_dataloader("./cv-parktikum/BSDS500-master/BSDS500/data")
    print(loader)