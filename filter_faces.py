import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from tqdm.auto import tqdm

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import clip
import shutil


final_dir = Path("dedup")
final_dir.mkdir(exist_ok=True)

input_dir = Path("aligned/apple-data")
sub_dirs = list(input_dir.glob("*/"))


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

all_count = 0
for movie_dir in tqdm(sub_dirs):
    sim_threshold = 0.96
    all_ims = list(movie_dir.rglob("*.png"))
    device = "cuda:0"
    model, preprocess = clip.load("ViT-B/32", device=device)

    feat = []
    ds = []
    if len(all_ims) == 0:
        continue
    for im_path in tqdm(all_ims):
        im = Image.open(im_path)
        with torch.no_grad():
            inp = preprocess(im).to(device).unsqueeze(0)
            feat.append(model.encode_image(inp))

        gray = np.array(im.convert("L"))
        gray = gray[4*64:4*192, 4*64:4*192]
        d = variance_of_laplacian(gray)
        ds.append(d)


    feat = torch.cat(feat, dim=0)
    feat_norm = feat/feat.norm(dim=1, keepdim=True)
    sims = (feat_norm @ feat_norm.T).cpu().numpy()

    avail = np.ones((len(all_ims), ))
    curr = 0
    ds = np.array(ds)
    count = 0
    while any(avail):
        current_sims = sims[curr]*avail

        idx = np.argwhere(current_sims > sim_threshold)

        this_ds = ds[idx]
        if len(this_ds) == 0:
            break
        avail[idx] = 0
        avail[:curr] = 0
        best = np.argmax(np.squeeze(this_ds))
        chosen = np.array(all_ims)[idx]
        chosen = chosen[best]
        chosen = chosen[0]
        if np.max(this_ds) > 2:
            # this_dir = final_dir/movie_dir.name
            this_dir = final_dir
            this_dir.mkdir(exist_ok=True)
            shutil.copy(chosen, this_dir/f"{all_count:06}.png")
            count += 1
            all_count += 1
            next_available = np.argwhere(avail==1)
            if len(next_available) > 0:
                curr = int(next_available[0])