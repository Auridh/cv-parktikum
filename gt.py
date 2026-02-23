import os
import numpy as np
import scipy.io as sio
from PIL import Image


if __name__ == "__main__":
    img_path = "./BSDS500-master/BSDS500/data/images/test"
    gt_path = "./BSDS500-master/BSDS500/data/groundTruth/test"
    out_path = "./predictions/gto"
    os.makedirs(out_path, exist_ok=True)

    # Get sorted lists of images and ground truths
    img_files = sorted([f for f in os.listdir(img_path) if f.endswith(".jpg")])
    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(".mat")])

    for img_file, gt_file in zip(img_files, gt_files):
        # Load image
        img = Image.open(os.path.join(img_path, img_file)).convert("RGB")
        img_np = np.array(img)

        # Load and combine ground truth masks
        muf = sio.loadmat(os.path.join(gt_path, gt_file))
        mu = muf.get("groundTruth")
        _, r = mu.shape
        masks = [np.array(mu[0, i]["Boundaries"][0, 0], dtype=bool) for i in range(r)]
        combined = np.zeros_like(masks[0], dtype=bool)
        for m in masks:
            combined |= m  # logical OR

        # Create an overlay: red where edge exists
        overlay = img_np.copy()
        overlay[combined] = [255, 0, 0]  # Red edges

        # Save the overlay image
        out_file = os.path.join(out_path, img_file.replace(".jpg", "_overlay.png"))
        Image.fromarray(overlay).save(out_file)
        print(f"Saved overlay: {out_file}")