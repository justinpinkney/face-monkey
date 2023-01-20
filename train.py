from random import shuffle
import torch
import torchvision
from tqdm import tqdm

from detect_faces import ImagePathDataset

net = torchvision.models.resnet50(pretrained=True)
tform = torchvision.transforms.Compose((
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
))
# pretrained = list([x for x in net.modules()])[:-1]
net.fc = torch.nn.Linear(2048, 2)
for p in net.parameters():
    p.requires_grad=True
# net = torch.nn.Sequential(*pretrained)
# print(net)

ds = torchvision.datasets.ImageFolder("labelled", transform=tform)

loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

net.train()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
device = "cuda:0"
net.to(device)
criteria = torch.nn.CrossEntropyLoss()

epochs = 10
for e in range(epochs):
    for data in tqdm(loader):
        opt.zero_grad()
        inp, labels = data
        pred = net(inp.to(device))
        loss = criteria(pred, labels.to(device))
        print(loss.item())
        loss.backward()
        opt.step()

from pathlib import Path
from PIL import Image
ext = ".jpg"

val_data = list(Path("dedup").glob(f"*{ext}"))
val_ds = ImagePathDataset(val_data, transform=tform, return_path=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)
import random
random.shuffle(val_data)

good_dir = Path("pred/good")
bad_dir = Path("pred/bad")
[x.mkdir(parents=True, exist_ok=True) for x in (good_dir, bad_dir)]
import shutil

net.eval()
count = 0
with torch.no_grad():
    for batch in tqdm(val_loader):
        paths, images = batch
        pred = net(images.to(device))
        results = pred.softmax(dim=1)
        for res, im_path in zip(results, paths):
            if res[0] > 0.5:
                shutil.copyfile(im_path, bad_dir/f"{count:08}{ext}")
            else:
                shutil.copyfile(im_path, good_dir/f"{count:08}{ext}")
            count += 1
