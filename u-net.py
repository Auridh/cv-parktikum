import os
import glob
import argparse
import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import f1_score


class ContourDataset(Dataset):
    def __init__(self, data_root, split):
        self.image_dir = os.path.join(data_root, "images", split)
        self.gt_dir = os.path.join(data_root, "groundTruth", split)

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))

        self.to_tensor = transforms.ToTensor()

    def load_mat_contours(self, mat_path):
        muf = sio.loadmat(mat_path)
        mu = muf.get("groundTruth")

        _, r = mu.shape # pyright: ignore[reportOptionalMemberAccess]
        masks = [
            np.array(mu[0, i]["Boundaries"][0, 0], dtype=np.uint8) # pyright: ignore[reportOptionalSubscript]
            for i in range(r)
        ]

        # Combine multiple boundaries into one mask
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = np.logical_or(combined, m)

        # majority vote
        #combined_mask = np.mean(masks, axis=0)
        #combined_mask = (combined_mask >= 0.5).astype(np.uint8)

        return combined.astype(np.uint8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        mat_path = os.path.join(self.gt_dir, filename + ".mat")

        image = Image.open(img_path).convert("RGB")
        image = self.to_tensor(image)

        mask = self.load_mat_contours(mat_path)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def _pad_to_match(self, x, ref):
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        return F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = self._pad_to_match(d4, e4)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self._pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self._pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self._pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.out(d1)


def compute_pos_weight(loader, device):
    total_edge = 0
    total_background = 0

    for imgs, masks in loader:
        masks = masks.to(device)
        masks_flat = masks.view(-1)
        total_edge += torch.sum(masks_flat == 1).item()
        total_background += torch.sum(masks_flat == 0).item()

    if total_edge == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(total_background / total_edge, device=device)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def weighted_bce_loss(pred, target, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(pred, target)

def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # compute positive weight dynamically
    pos_weight = compute_pos_weight(train_loader, device)
    print(f"Using pos_weight={pos_weight.item():.2f} for weighted BCE")

    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = weighted_bce_loss(outputs, masks, pos_weight) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, device, pos_weight)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model_unet.pth")
            print("Saved best model")

    print("Training complete.")

@torch.no_grad()
def evaluate(model, val_loader, device, pos_weight):
    model.eval()
    total_loss = 0
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = weighted_bce_loss(outputs, masks, pos_weight) + dice_loss(outputs, masks)
        total_loss += loss.item()
    return total_loss / len(val_loader)

@torch.no_grad()
def predict(model, dataset, loader, device, output_dir="predictions", threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_f1 = []

    for i, (img, mask) in enumerate(loader):
        img, mask = img.to(device), mask.to(device)
        output = torch.sigmoid(model(img))

        # original filename
        original_path = dataset.image_paths[i]
        name, _ = os.path.splitext(os.path.basename(original_path))
        print("#"*40)
        print(f"# {os.path.basename(original_path)}")

        # Save probability map
        prob_map = (output[0, 0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(prob_map).save(os.path.join(output_dir, f"unet_prob_{name}.png"))

        # Compute thresholded prediction
        pred = (output > threshold).cpu().numpy().astype(np.uint8)

        # Save thresholded mask
        mask_img = (pred[0, 0] * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(os.path.join(output_dir, f"unet_binary_{name}.png"))

        # Compute F1 score
        mask_np = mask.cpu().numpy().astype(np.uint8)
        f1 = f1_score(mask_np.flatten(), pred.flatten())
        all_f1.append(f1)

        print(f"> F1: {f1:.4f}")

    mean_f1 = np.mean(all_f1)
    print(f"Mean F1 Score: {mean_f1:.4f}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ContourDataset(args.data_root, "train")
    val_dataset = ContourDataset(args.data_root, "val")
    test_dataset = ContourDataset(args.data_root, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=1, base=32).to(device)

    if not args.predict:
        train_model(model, train_loader, val_loader, device, args.epochs, args.lr)

    model.load_state_dict(torch.load("best_model_unet.pth"))
    predict(model, test_dataset, test_loader, device, "predictions/unet", threshold=args.threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./BSDS500-master/BSDS500/data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--predict", default=False, action='store_true')
    parser.add_argument("--threshold", default=0.8, type=float)

    args = parser.parse_args()
    main(args)
