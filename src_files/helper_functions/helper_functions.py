import os
import csv
from copy import deepcopy, copy
import random
import math
import numpy as np
import torch
import torch.optim as optim

from PIL import Image, ImageDraw
from torchvision import datasets as datasets
from pycocotools.coco import COCO
import json
import torch.utils.data as data
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd


def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01

class MultiLabelCelebA(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path: str, 
        split: str = "train",
        transform = None,
        target_transform=None
        ) -> None:
        self.data_path = data_path
        self.split = split
        if not os.path.exists(data_path):
            raise RuntimeError("Folder with data for CELEBA not exists")
        
        keep = []
        with open(os.path.join(data_path, "list_eval_partition.txt")) as f:
            lines = f.readlines()
            for line in lines:
                filename, numb = line.split()
                if split == "train" and int(numb) == 0:
                    keep.append(filename)
                elif split == "valid" and int(numb) == 1:
                    keep.append(filename)
                elif split == "test" and int(numb) == 2:
                    keep.append(filename)
                elif split == "all":
                    keep.append(filename)
        
        self.label_names, self.labels = self._load_labels(os.path.join(data_path, "list_attr_celeba.txt"), keep)
        self.file_names = list(self.labels.keys())
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_labels(self, path: str, keep = None):
        row_names = []
        row_data_dict = {}
        with open(path, 'r') as file:
            reader = csv.reader(file)
            line_numb = 0
            for row in reader:
                row_data = row[0]
                if line_numb == 1:
                    row_names = row_data.split()
                elif line_numb > 1:
                    row_data_numb = row_data.split()
                    filename = row_data_numb[0]
                    if keep is not None:
                        if filename in keep:
                            row_data_values = [int(x) if int(x) == 1 else int(x)+1 for x in row_data_numb[1:]]
                            row_data_dict[filename] = row_data_values
                    else:
                        row_data_values = [int(x) if int(x) == 1 else int(x)+1 for x in row_data_numb[1:]]
                        row_data_dict[filename] = row_data_values

                line_numb += 1
                
        return row_names, row_data_dict
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, "img_align_celeba", self.file_names[index])).convert('RGB')
        label = self.labels[self.file_names[index]]
        target = torch.zeros((3, 40), dtype=torch.long)
        target[0] = torch.tensor(label, dtype=torch.long)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class VOCDataset(datasets.coco.CocoDetection):
    def __init__(self, root, transform=None, target_transform=None, val=False, boxcrop=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.boxcrop = boxcrop
        if val:
            image_sets_file = os.path.join(self.root,"ImageSets/Main/train.txt")
        else:
            image_sets_file = os.path.join(self.root,"ImageSets/Main/val.txt")
        self.ids = VOCDataset._read_image_ids(image_sets_file)

        # if the labels file exists, read in the class names
        self.class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        target = torch.zeros((3, len(self.class_names)), dtype=torch.long)
        target[0] = torch.tensor(labels, dtype=torch.long)
        img = Image.open(os.path.join(self.root, f"JPEGImages/{image_id}.jpg")).convert('RGB')
        
        if self.boxcrop:
            img = crop_box(img, boxes, self.boxcrop, cut_img=0.2)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.root, f"Annotations/{image_id}.xml")
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = [0 for x in range(len(self.class_names))]
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels[self.class_dict[class_name]] = 1
        return boxes, labels
    
    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, boxcrop=None):
        self.root = root
        self.coco = COCO(annFile)
        self.boxcrop = boxcrop
        
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        boxes = [x['bbox'] for x in target]

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.boxcrop:
            img = crop_box(img, boxes, self.boxcrop, cut_img=0.15)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_class_ids_split(json_path, classes_dict):
    with open(json_path) as fp:
        split_dict = json.load(fp)
    if 'train class' in split_dict:
        only_test_classes = False
    else:
        only_test_classes = True

    train_cls_ids = set()
    val_cls_ids = set()
    test_cls_ids = set()

    # classes_dict = self.learn.dbunch.dataset.classes
    for idx, (i, current_class) in enumerate(classes_dict.items()):
        if only_test_classes:  # base the division only on test classes
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)
            else:
                val_cls_ids.add(idx)
                train_cls_ids.add(idx)
        else:  # per set classes are provided
            if current_class in split_dict['train class']:
                train_cls_ids.add(idx)
            # if current_class in split_dict['validation class']:
            #     val_cls_ids.add(i)
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)

    train_cls_ids = np.fromiter(train_cls_ids, np.int32)
    val_cls_ids = np.fromiter(val_cls_ids, np.int32)
    test_cls_ids = np.fromiter(test_cls_ids, np.int32)
    return train_cls_ids, val_cls_ids, test_cls_ids


def update_wordvecs(model, train_wordvecs=None, test_wordvecs=None):
    if hasattr(model, 'fc'):
        if train_wordvecs is not None:
            model.fc.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.fc.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    elif hasattr(model, 'head'):
        if train_wordvecs is not None:
            model.head.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.head.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    else:
        print("model is not suited for ml-decoder")
        exit(-1)


def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')
    # return Image.open(path).convert('RGB')

class DatasetFromList(data.Dataset):
    """From List dataset."""

    def __init__(self, root, impaths, labels, idx_to_class,
                 transform=None, target_transform=None, class_ids=None,
                 loader=default_loader):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.classes = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = tuple(zip(impaths, labels))
        self.class_ids = class_ids

    def __getitem__(self, index):
        impath, target = self.samples[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform([target])
        target = self.get_targets_multi_label(np.array(target))
        if self.class_ids is not None:
            target = target[self.class_ids]
        return img, target

    def __len__(self):
        return len(self.samples)

    def get_targets_multi_label(self, target):
        # Full (non-partial) labels
        labels = np.zeros(len(self.classes))
        labels[target] = 1
        target = labels.astype('float32')
        return target

def parse_csv_data(dataset_local_path, metadata_local_path):
    try:
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    except FileNotFoundError:
        # No data.csv in metadata_path. Try dataset_local_path:
        metadata_local_path = dataset_local_path
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    images_path_list = df.values[:, 0]
    # images_path_list = [os.path.join(dataset_local_path, images_path_list[i]) for i in range(len(images_path_list))]
    labels = df.values[:, 1]
    image_labels_list = [labels.replace('[', "").replace(']', "").split(', ') for labels in
                             labels]

    if df.values.shape[1] == 3:  # split provided
        valid_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'val']
        train_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'train']
    else:
        valid_idx = None
        train_idx = None

    # logger.info("em: end parsr_csv_data: num_labeles: %d " % len(image_labels_list))
    # logger.info("em: end parsr_csv_data: : %d " % len(image_labels_list))

    return images_path_list, image_labels_list, train_idx, valid_idx


def multilabel2numeric(multilabels):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(multilabels)
    classes = multilabel_binarizer.classes_
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    multilabels_numeric = []
    for multilabel in multilabels:
        labels = [class_to_idx[label] for label in multilabel]
        multilabels_numeric.append(labels)
    return multilabels_numeric, class_to_idx, idx_to_class


def get_datasets_from_csv(dataset_local_path, metadata_local_path, train_transform,
                          val_transform, json_path):

    images_path_list, image_labels_list, train_idx, valid_idx = parse_csv_data(dataset_local_path, metadata_local_path)
    labels, class_to_idx, idx_to_class = multilabel2numeric(image_labels_list)

    images_path_list_train = [images_path_list[idx] for idx in train_idx]
    image_labels_list_train = [labels[idx] for idx in train_idx]

    images_path_list_val = [images_path_list[idx] for idx in valid_idx]
    image_labels_list_val = [labels[idx] for idx in valid_idx]

    train_cls_ids, _, test_cls_ids = get_class_ids_split(json_path, idx_to_class)

    train_dl = DatasetFromList(dataset_local_path, images_path_list_train, image_labels_list_train,
                               idx_to_class,
                               transform=train_transform, class_ids=train_cls_ids)

    val_dl = DatasetFromList(dataset_local_path, images_path_list_val, image_labels_list_val, idx_to_class,
                             transform=val_transform, class_ids=test_cls_ids)

    return train_dl, val_dl, train_cls_ids, test_cls_ids

def crop_box(img, boxes, size, scale=(0.2, 0.2), cut_img=0.1):
    width, height = img.size
    max_height = height * scale[0]
    max_width = width * scale[1]
    crop_height_up = float(torch.rand(1) * max_height)
    crop_width_left = float(torch.rand(1) * max_width)
    crop_height_down = float(torch.rand(1) * max_height)
    crop_width_right = float(torch.rand(1) * max_width)
    for box in boxes:
        x1, y1, x2, y2 = box
        if crop_width_left > x1:
            if crop_width_left-x1 > (x2-x1)*cut_img:
                crop_width_left = x1 + (x2-x1)*cut_img
        if width-crop_width_right < x2:
            if x2-(width-crop_width_right) > (x2-x1)*cut_img:
                crop_width_right = width - (x2-(x2-x1)*cut_img)
        if height-crop_height_down < y2:
            if y2-(height-crop_height_down) > (y2-y1)*cut_img:
                crop_height_down = height - (y2-(y2-y1)*cut_img)
        if crop_height_up > y1:
            if crop_height_up-y1 > (y2-y1)*cut_img:
                crop_height_up = y1 + (y2-y1)*cut_img

    cropped_img = img.crop((int(crop_width_left),int(crop_height_up),int(width-crop_width_right),int(height-crop_height_down)))
    resized_img = cropped_img.resize((size, size))
    return resized_img

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate_con
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs_con)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr