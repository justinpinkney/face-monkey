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

from alignment import align_face
from concurrent.futures import ThreadPoolExecutor

sys.path.append('yolov5-face')
from models.experimental import attempt_load
from utils.general import non_max_suppression_face
from joblib import Parallel, delayed

import av

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename) -> None:
        super().__init__()
        self.filename = filename
        self.container = av.open(filename)
        self.container.streams.video[0].thread_type = "AUTO"
    def __iter__(self):
        return self.get_frame()

    def get_frame(self):
        return (frame.to_ndarray(format="rgb24") for frame in self.container.decode(video=0))

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


@torch.no_grad()
def process_dir(im_dir, model):
    ims = sorted(list(Path(im_dir).rglob("*.jpg")))
    tforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    ds = ImagePathDataset(ims, transform=tforms, return_path=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=n_workers)

    has_face = []
    for batch in tqdm(loader):
        im = batch
        im = im.to(device)
        pred = model(im)[0]
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        has_face.extend([x.shape[0] > 0 for x in pred])

        for xs, this_i in zip(pred, batch):
            np_im = this_i.permute(1,2,0).cpu().numpy().copy()
            for x in xs:
                q = x[:4].cpu().numpy()
                x1, y1, x2, y2 = [int(el) for el in q]
                print(x.shape)

                np_im = cv2.rectangle(np_im, (x1,y1), (x2, y2), (0,1,0))

            im = Image.fromarray((255*np_im).astype(np.uint8)).save("rean.jpg")

@torch.no_grad()
def process_video(filename, out_dir=Path("frames")):
    out_dir.mkdir(exist_ok=True)
    bs = 64
    count = 0
    h_threshold = 256
    ratio_thresh = 0.5
    target_w = 15*32
    n_workers = 32
    executor = ThreadPoolExecutor(max_workers=n_workers)

    ds = VideoDataset(filename)
    loader = torch.utils.data.DataLoader(ds, num_workers=1, batch_size=bs)
    for frames in loader:
        _, orig_h, orig_w, _ = frames.shape
        ratio = target_w/orig_w
        target_h = int(orig_h*ratio/32)*32

        frames = frames.to(device).permute(0, 3, 1, 2).float()/255
        inp = torch.nn.functional.interpolate(frames, (target_h, target_w), mode="area")
        # cv2.imwrite("dump.jpg", 255*inp[0].cpu().permute(1,2,0).numpy())
        pred = model(inp)[0]
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        pred = [p.cpu().numpy() for p in pred]
        frames = frames.cpu().numpy()
        fnames = [out_dir/f"{count+x:06}.png" for x in range(len(frames))]
        for fname, frame, p in zip(fnames, frames, pred):
            process_pred(p, fname, h_threshold, ratio_thresh, frame, orig_h, orig_w, target_h, target_w, executor)
        count += bs
    executor.shutdown(wait=True)
    print("done")


def process_pred(pred, fname, h_threshold, ratio_thresh, frame, orig_h, orig_w, target_h, target_w, executor):

    for idx, x in enumerate(pred):
        q = x[:4]
        x1, y1, x2, y2 = [int(el) for el in q]

        convert_x = lambda x: x*orig_w/target_w
        convert_y = lambda y: y*orig_h/target_h

        x1 = convert_x(x1)
        x2 = convert_x(x2)
        y1 = convert_y(y1)
        y2 = convert_y(y2)
        face_h = np.abs(y1-y2)
        lm = x[5:]
        eye_left = np.array(((convert_x(lm[0]), convert_y(lm[1]))))
        eye_right = np.array(((convert_x(lm[2]), convert_y(lm[3]))))
        lm_mouth_outer = np.array(
                    ((convert_x(lm[6]), convert_y(lm[7])),
                    (convert_x(lm[8]), convert_y(lm[9])))
                    )


        if face_h < h_threshold:
            continue
        edge_thresh = 50
        if any([x < edge_thresh for x in eye_left]) or \
            any([x < edge_thresh for x in eye_right]):
            continue

        inter_eye = np.linalg.norm(eye_left - eye_right)
        vec = (eye_left + eye_right)/2
        vec = vec - (lm_mouth_outer[0] + lm_mouth_outer[1])/2
        eye_mouth = np.linalg.norm(vec)
        if inter_eye/eye_mouth > ratio_thresh:
            executor.submit(do_crops,fname, frame, lm_mouth_outer, eye_left, eye_right, idx)

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def do_crops(fname, frame, lm_mouth_outer, eye_left, eye_right, idx, var_thresh=5):
    fname = Path(fname)
    img_in = Image.fromarray((255*frame.transpose(1,2,0)).astype(np.uint8))
    im = align_face(img_in, lm_mouth_outer, eye_left, eye_right).convert("RGB")
    var = variance_of_laplacian(np.array(im))
    if var > var_thresh:
        im.save(fname.parent / f"{fname.stem}-{idx}.jpg", quality=95)

if __name__ == "__main__":
    video_dir = Path("apple-data")
    aligned_dir = Path("aligned")
    aligned_dir.mkdir(exist_ok=True)
    import sys

    dev_idx = sys.argv[1]
    device = f"cuda:{dev_idx}"
    print(device)
    model_weights = "pretrained/yolov5-face-l.pt"
    bs = 2
    conf_thres = 0.5
    iou_thres = 0.5

    model = attempt_load(model_weights, map_location="cpu").to(device)
    # process_dir(im_dir, model)

    videos = list(video_dir.rglob("*.mov"))
    for f in tqdm(videos):
        this_dir = aligned_dir/f.parent/f.stem
        if this_dir.exists():
            print(f"skipping {this_dir}")
        else:
            this_dir.mkdir(parents=True)
            print(f"processing {this_dir}")
            try:
                process_video(str(f), out_dir=this_dir)
            except Exception as e:
                print(f"failed on {this_dir}")
                print(e)
    print("Finished, waiting")