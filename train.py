from random import shuffle
import torch
import torchvision
from tqdm import tqdm

net = torchvision.models.resnet50(pretrained=True)
tform = torchvision.transforms.Compose((
    torchvision.transforms.Resize((224,224)),
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

epochs = 5
for e in range(epochs):
    for data in tqdm(loader):
        opt.zero_grad()
        inp, labels = data
        pred = net(inp.to(device))
        loss = criteria(pred, labels.to(device))
        loss.backward()
        opt.step()

from pathlib import Path
from PIL import Image
val_data = sorted(list(Path("dedup").glob("*.png")))

good_dir = Path("pred/good")
bad_dir = Path("pred/bad")
[x.mkdir(parents=True, exist_ok=True) for x in (good_dir, bad_dir)]
import shutil

net.eval()
for idx, v in tqdm(enumerate(val_data)):
    im = Image.open(v)
    inp = tform(im)
    pred = net(inp.to(device).unsqueeze(0))
    res = pred.softmax(dim=1)
    if res[0][0] > 0.5:
        shutil.copyfile(v, bad_dir/f"{idx:06}.png")
    else:
        shutil.copyfile(v, good_dir/f"{idx:06}.png")
