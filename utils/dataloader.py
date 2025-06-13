import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

# ----------- Add this helper function to map RGB -> Class index ------------
def rgb_to_label(rgb_img):
    """Convert RGB label image to 2D label indices."""
    colormap = {
        (0, 0, 0): 0,         # Background
        (0, 0, 255): 1,       # Human divers
        (0, 255, 0): 2,       # Plants
        (0, 255, 255): 3,     # Wrecks
        (255, 0, 0): 4,       # Robots
        (255, 0, 255): 5,     # Reefs
        (255, 255, 0): 6,     # Fish
        (255, 255, 255): 7,   # Sea-floor
    }

    label_map = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
    for rgb, idx in colormap.items():
        mask = np.all(rgb_img == rgb, axis=-1)
        label_map[mask] = idx
    return label_map


class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # --- Load RGB image and label mask (in RGB format) ---
        jpg = Image.open(os.path.join(self.dataset_path, "SUIM2022/JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.dataset_path, "SUIM2022/SegmentationClass", name + ".png")).convert("RGB")

        # --- Data augmentation ---
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        # --- Convert RGB label image to class index map ---
        png = rgb_to_label(np.array(png))

        # --- Convert input image ---
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])

        # --- Clip to valid class range just in case ---
        png[png >= self.num_classes] = self.num_classes

        # --- One-hot encode segmentation labels ---
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))  # Already RGB, will be processed later
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('RGB', (w, h), (0, 0, 0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # --- Random scaling and aspect ratio jitter ---
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # --- Horizontal flip ---
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # --- Place image ---
        dx, dy = int(self.rand(0, w - nw)), int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('RGB', (w, h), (0, 0, 0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        # --- Blur ---
        if self.rand() < 0.25:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # --- Rotation ---
        if self.rand() < 0.25:
            center = (w // 2, h // 2)
            angle = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
            label = Image.fromarray(label)

        # --- Color jitter in HSV ---
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        h_, s_, v_ = cv2.split(hsv)
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=np.uint8)
        lut_h = ((x * r[0]) % 180).astype(dtype)
        lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((
            cv2.LUT(h_, lut_h),
            cv2.LUT(s_, lut_s),
            cv2.LUT(v_, lut_v)
        ))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def deeplab_dataset_collate(batch):
    images, pngs, seg_labels = zip(*batch)
    images = torch.from_numpy(np.array(images)).float()
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).float()
    return images, pngs, seg_labels



# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
