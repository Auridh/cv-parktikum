import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import scipy.io as sio
from PIL import Image
import os
import sys
from tqdm import tqdm


def feature_extraction(file):
    # Load image this time with the direct pillow function
    # convert('L') converts to grayscale
    img = Image.open(file).convert("L")
    # this could be better, because smaller values are commonly better for network training????
    img = np.array(img) / 255.0
    H, W = img.shape
    # pytorch often expects shapes in the form (N, C, H, W)
    # N = batch size
    # C = number of channels
    # H = height
    # W = width
    # with unsqueeze we get from our initial shape of (H,W) a shape of (1,1,H,W)
    # it now fits the common pytorch shape convention
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    # Sobel strength
    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    img_tensor = F.pad(img_tensor, (1, 1, 1, 1), mode='reflect')
    Gx = F.conv2d(img_tensor, sobel_x)
    Gy = F.conv2d(img_tensor, sobel_y)

    # it could be necessary to add a small factor to avoid null division on back propagation
    # like + 1e-8
    strength = torch.sqrt(Gx**2 + Gy**2)
    strength = strength / strength.max()
    #strength = (strength - strength.mean()) / (strength.std() + 1e-8)

    # create two features each containing the normalized x or y coordinate
    x_coord = []
    y_coord = []
    for h in range(H):
        x_row = []
        y_row = []
        y_entry = h / H
        for w in range(W):
            x_row.append(w / W)
            y_row.append(y_entry)
        x_coord.append(x_row)
        y_coord.append(y_row)

    x_coord = (
        torch.tensor(x_coord, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )
    y_coord = (
        torch.tensor(y_coord, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )

    # first reorder the tensor with permute then reshape/flatten it
    # (1,1,H,W) -> (1,H,W,1) -> (H*W,1)
    # each row is one pixel
    # each column is one feature
    strength = strength.permute(0, 2, 3, 1).reshape(-1, 1)
    #x_coord = x_coord.permute(0, 2, 3, 1).reshape(-1, 1)
    #y_coord = y_coord.permute(0, 2, 3, 1).reshape(-1, 1)
    #features = torch.cat((strength, x_coord, y_coord), 1)
    features = torch.cat((strength,), 1)

    return features, (H, W)

def label_extraction(file):
    muf = sio.loadmat(file)
    mu = muf.get("groundTruth")
    _, r = mu.shape

    masks = [
        np.array(mu[0, i]["Boundaries"][0, 0], dtype=np.uint8)
        for i in range(r)
    ]

    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        combined = np.logical_or(combined, m)

    combined = combined.astype(np.float32)

    labels = torch.from_numpy(combined).reshape(-1, 1).to(device)
    return labels

class EdgeMLP(nn.Module):
    def __init__(
        self, input_dim=3, hidden_dim_input=32, hidden_dim_output=16, output_dim=1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_input),
            nn.ReLU(),
            nn.Linear(hidden_dim_input, hidden_dim_output),
            nn.ReLU(),
            nn.Linear(hidden_dim_output, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_per_image(features, labels, shape, epochs=20, lr=0.001, threshold=None):
    N = features.shape[0]
    
    idx = torch.randperm(N)
    train_size = int(0.1 * N)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]

    model = EdgeMLP(features.shape[1]).to(device)

    pos = torch.sum(y_train == 1)
    neg = torch.sum(y_train == 0)
    pos_weight = (neg / pos).clamp(min=1.0)
    print(f"> PW: {pos_weight}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        # test F1 remaining 90 percent
        logits_test = model(X_test)
        probs_test = torch.sigmoid(logits_test)
        preds_test = (probs_test > 0.5).float()
        f1 = f1_score(
            y_test.cpu().numpy(),
            preds_test.cpu().numpy()
        )

        # full image prediction
        logits_full = model(features)
        probs_full = torch.sigmoid(logits_full)

        # adaptive threshold
        if threshold is None:
            y_true = labels.cpu().numpy()
            prob_vals = probs_full.cpu().numpy()

            best_f1 = 0
            best_t = 0.5
            for t in np.linspace(0.3, 0.7, 40):
                pred_bin = (prob_vals > t).astype(np.float32)
                f1_t = f1_score(y_true, pred_bin)
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_t = t
            threshold = best_t
            print(f"> OT: {threshold:.3f}")

        preds_full = (probs_full > threshold).float()

    return f1, preds_full, probs_full


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    in_path = "./BSDS500-master/BSDS500/data"
    out_path = "./predictions/mlp"
    test_images = os.path.join(in_path, "images", "test")
    test_gt = os.path.join(in_path, "groundTruth", "test")

    os.makedirs(out_path, exist_ok=True)

    image_files = sorted(
        [f for f in os.listdir(test_images) if f.endswith(".jpg")]
    )
    gt_files = sorted(
        [f for f in os.listdir(test_gt) if f.endswith(".mat")]
    )

    all_f1 = []
    all_probs_min = []
    all_probs_max = []
    all_probs_mean = []

    for img_file, gt_file in zip(image_files, gt_files):
        print("#"*40)
        print(f"# {img_file}")

        features, shape = feature_extraction(
            os.path.join(test_images, img_file)
        )
        labels = label_extraction(
            os.path.join(test_gt, gt_file)
        )

        f1, preds_full, probs_full = train_per_image(
            features, labels, shape, epochs=100
        )

        all_f1.append(f1)
        print(f"> F1: {f1}")
        all_probs_min.append(probs_full.min().item())
        all_probs_max.append(probs_full.max().item())
        all_probs_mean.append(probs_full.mean().item())

        H, W = shape

        pred_img = preds_full.cpu().numpy().reshape(H, W)
        pred_img = (pred_img * 255).astype(np.uint8)
        Image.fromarray(pred_img).save(
            os.path.join(out_path, f"mlp_binary_{img_file}")
        )

        prob_img = probs_full.cpu().numpy().reshape(H, W)
        prob_img = (prob_img * 255).astype(np.uint8)
        Image.fromarray(prob_img).save(
            os.path.join(out_path, f"mlp_prob_{img_file}")
        )

    print("\n====================")
    print("Mean F1 over test set:", np.mean(all_f1))
    print("Mean Min Prob over test set:", np.mean(all_probs_min))
    print("Mean Max Prob over test set:", np.mean(all_probs_max))
    print("Mean Prob over test set:", np.mean(all_probs_mean))
