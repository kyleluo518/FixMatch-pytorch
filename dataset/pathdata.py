import logging
import math
import os
import csv

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from randaugment import RandAugmentMC

logger = logging.getLogger(__name__)
imagesize = 96

path_mean = (0.5, 0.5, 0.5)
path_std = (0.5, 0.5, 0.5)

def tensorfy_data(path):
    imagepath = os.path.join(path, "images")
    valpath = os.path.join(path, "val.csv")
    # read through images and store in TensorDataset
    images = {}
    with os.scandir(imagepath) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                img = Image.open(entry.path)
                images[entry.name] = [transforms.functional.pil_to_tensor(img)]
    with open(valpath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['filename'] in images:
                raise LookupError('Labels not correct')
            images[row['filename']].append(row['label'])
    samples, targets = [], []
    for filename in images:
        image, label = images[filename]
        samples.append(image)
        targets.append(label)
    base_dataset = torch.utils.data.TensorDataset(torch.stack(samples), torch.stack(targets))
    return base_dataset


def get_path(args, path):
    base_dataset = tensorfy_data(os.path.join(path, "Unlabeled")
    # Transformations
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=imagesize,
                              padding=int(imagesize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=path_mean, std=path_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=path_mean, std=path_std)
    ])
    
    # Index split
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)



def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=imagesize,
                                  padding=int(imagesize*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=imagesize,
                                  padding=int(imagesize*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def main():
    get_path(None, '/data/PathData/PathData/Pcam/Unlabeled/images/', '/data/PathData/PathData/Pcam/Unlabeled/images/')

if __name__ == '__main__':
    main()

DATASET_GETTERS = {'path': get_path}
