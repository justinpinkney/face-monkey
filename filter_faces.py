from locale import normalize
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

class ImagePathDataset(Dataset):
    """Simple dataset for loading samples from image paths"""
    def __init__(self, im_paths, transform=None, return_path=False) -> None:
        super().__init__()
        self.image_files = im_paths
        self.transform = transform
        self.return_path = return_path

    def __getitem__(self, index):
        img = Image.open(self.image_files[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.return_path:
            im_path = str(self.image_files[index])
            return im_path, img
        else:
            return img

    def __len__(self):
        return len(self.image_files)

import kornia
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	# return cv2.Laplacian(image, cv2.CV_64F).var()
    return kornia.filters.laplacian(image, 3, normalized=False).var(dim=[2,3])

all_count = 0
bs = 64

n_px = 224
clip_preporc = transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

device = "cuda:0"
model, preprocess = clip.load("ViT-B/32", device=device)
sim_threshold = 0.96

for movie_dir in tqdm(sub_dirs):
    all_ims = list(movie_dir.rglob("*.jpg"))

    feat = []
    ds = []
    if len(all_ims) == 0:
        continue

    dataset = ImagePathDataset(all_ims, transform=transforms.ToTensor())
    loader = DataLoader(dataset, bs, num_workers=8)

    print(f"processing {movie_dir}")
    for batch in tqdm(loader):
        with torch.no_grad():
            batch = batch.to(device)
            inp = clip_preporc(batch)
            feat.append(model.encode_image(inp))

        # gray = np.array(im.convert("L"))
            gray = batch.mean(1, keepdim=True)
            gray = 255*gray[:, :, 4*64:4*192, 4*64:4*192]
            d = variance_of_laplacian(gray).cpu()
        ds.append(d)

    print(f"calculated metrics")

    feat = torch.cat(feat, dim=0)
    feat_norm = feat/feat.norm(dim=1, keepdim=True)
    sims = (feat_norm @ feat_norm.T).cpu().numpy()

    avail = np.ones((len(all_ims), ))
    curr = 0
    ds = torch.cat(ds, dim=0)
    ds = np.array(ds)
    count = 0
    print(f"processing frames")
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
            shutil.copy(chosen, this_dir/f"{all_count:06}.jpg")
            count += 1
            all_count += 1
            next_available = np.argwhere(avail==1)
            if len(next_available) > 0:
                curr = int(next_available[0])
