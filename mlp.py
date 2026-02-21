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

    Gx = F.conv2d(img_tensor, sobel_x, padding=1)
    Gy = F.conv2d(img_tensor, sobel_y, padding=1)

    # it could be necessary to add a small factor to avoid null division on back propagation
    # like + 1e-8
    strength = torch.sqrt(Gx**2 + Gy**2)
    strength = strength / strength.max()

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
    x_coord = x_coord.permute(0, 2, 3, 1).reshape(-1, 1)
    y_coord = y_coord.permute(0, 2, 3, 1).reshape(-1, 1)
    features = torch.cat((strength, x_coord, y_coord), 1)

    return features, (H, W)


def label_extraction(file):
    # Load groundTruths
    # I read that it's common to average in our case, so...
    muf = sio.loadmat(file)
    mu = muf.get("groundTruth")
    _, r = mu.shape

    boundaries = []
    for i in range(r):
        boundary = mu[0, i]["Boundaries"][0, 0]
        boundary = np.array(boundary)
        boundaries.append(boundary)

    avg_boundary = np.mean(boundaries, axis=0)
    # only haven true and false for edges simplifies the rest of the model
    binary_labels = (avg_boundary > 0.5).astype(np.float32)
    labels = torch.from_numpy(binary_labels).reshape(-1, 1).to(device)

    return labels


class EdgeDataset(Dataset):
    def __init__(self, train_path, test_path):
        self.X_train_path = os.path.abspath(train_path)
        self.X_test_path = os.path.abspath(test_path)
        self.X_train_files = sorted(
            [f for f in os.listdir(self.X_train_path) if f.lower().endswith(".jpg")]
        )
        self.X_test_files = sorted(
            [f for f in os.listdir(self.X_test_path) if f.lower().endswith(".mat")]
        )

    def __len__(self):
        return len(self.X_train_files)

    def __getitem__(self, idx):
        train_file = self.X_train_files[idx]
        test_file = self.X_test_files[idx]

        features, shape = feature_extraction(
            os.path.join(self.X_train_path, train_file)
        )
        labels = label_extraction(os.path.join(self.X_test_path, test_file))

        return features, labels, shape, train_file


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


def evaluate(model, loader, threshold=0.5, save_path=None):
    # we could compute the optimal threshold automatically??

    if save_path is not None:
        save_path = os.path.abspath(save_path)

    # enter evaluation mode
    model.eval()

    f1_scores = []
    # we don't need gradient calculation during evaluation
    with torch.no_grad():
        for features, labels, shape, (file,) in loader:
            # due to our batch size of 1, the shape of features is (1, H*W, 1)
            # now we pop the first dim to get (H*W, 1) again
            features = features.squeeze(0)
            labels = labels.squeeze(0)

            output = model(features)
            # sigmoid converts to probabilities [0,1] then we convert to binary 0 or 1 output
            # .float() since we need the same datatype everywhere
            predictions = (torch.sigmoid(output) > threshold).float()

            f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())
            f1_scores.append(f1)

            # print("Mean prediction:", predictions.mean().item())
            # print("Mean label:", labels.float().mean().item())

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)

                # get image size
                H, W = shape
                pred_img = predictions.cpu().numpy().reshape(H, W)
                pred_img = (pred_img * 255).astype(np.uint8)
                Image.fromarray(pred_img).save(os.path.join(save_path, f"pred_{file}"))

    return sum(f1_scores) / len(f1_scores)


def compute_pos_weight(loader):
    total_edge = 0
    total_background = 0

    for features, labels, _, _ in loader:
        # each pixel is now a separate value
        labels = labels.view(-1)
        # count
        total_edge += torch.sum(labels == 1).item()
        total_background += torch.sum(labels == 0).item()

    # relation between background and edge pixels
    return torch.tensor(total_background / total_edge).to(device)


def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    # pos_weight, because we have such a large difference in number between background and edge pixels
    pos_weight = compute_pos_weight(train_loader)
    # sigmoid + BCE
    bce_ll = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = 0

    # switch into training mode
    model.train()
    for epoch in range(epochs):

        for features, labels, _, _ in train_loader:
            features = features.squeeze(0)
            labels = labels.squeeze(0)

            # clear previous gradients
            optimizer.zero_grad()

            outputs = model(features)
            loss = bce_ll(outputs, labels)

            # compute gradients
            loss.backward()
            # update model
            optimizer.step()

        val_f1 = evaluate(model, val_loader)
        print(f"Epoch {epoch} | F1: {val_f1:.4f}")

        # save the best model state
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")

    return model


if __name__ == "__main__":
    # hardware acceleration if available
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    in_path = "./BSDS500-master/BSDS500/data"
    out_path = "./output"

    # get path from cmd
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        in_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    # all datasets
    train_dataset = EdgeDataset(
        os.path.join(in_path, "images", "train"),
        os.path.join(in_path, "groundTruth", "train"),
    )
    val_dataset = EdgeDataset(
        os.path.join(in_path, "images", "val"),
        os.path.join(in_path, "groundTruth", "val"),
    )
    test_dataset = EdgeDataset(
        os.path.join(in_path, "images", "test"),
        os.path.join(in_path, "groundTruth", "test"),
    )

    # shuffle??
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = EdgeMLP().to(device)
    model = train_model(
        model, train_loader, val_loader, epochs=100, learning_rate=0.001
    )

    # we saved the best working state
    model.load_state_dict(torch.load("best_model.pt"))

    # evaluate our model
    test_f1 = evaluate(model, test_loader, threshold=0.5, save_path=out_path)
    print("Final Test F1:", test_f1)
