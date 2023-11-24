'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import torch
import warnings
import pickle

import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc

def normalization(point):
    max = np.abs(point).max()
    # max = np.max(np.sqrt(np.sum(point ** 2, axis=1)))
    point = point / max
    return point

# 평행이동
def translation_xyz(point):
    x_max = np.max(point[:, 0])
    y_max = np.max(point[:, 1])
    z_max = np.max(point[:, 2])
    x_min = np.min(point[:, 0])
    y_min = np.min(point[:, 1])
    z_min = np.min(point[:, 2])

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    translation_x = point[:, 0] - x_center
    translation_y = point[:, 1] - y_center
    translation_z = point[:, 2] - z_center

    point[:, 0] = translation_x
    point[:, 1] = translation_y
    point[:, 2] = translation_z

    return point

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def collate_fn(list_data):
    batched_pts_list = []
    batched_labels_list = []
    for data_dict in list_data:
        pts = data_dict['pts']
        gt_labels = data_dict['target']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_labels_list.append(torch.from_numpy(gt_labels))

    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_labels=batched_labels_list
    )

    return rt_data_dict

def get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=False):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate,
    )
    return dataloader

class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.DSR = args.Delivery_Service_Robot_Dataset

        if self.DSR:
            self.catfile = os.path.join(self.root, 'Delivery_Service_Robot-shape_name.txt')
        else:
            if self.num_category == 8:
                self.catfile = os.path.join(self.root, 'mymodelnet8_shape_names.txt')
            elif self.num_category == 10:
                self.catfile = os.path.join(self.root, 'mymodelnet10_shape_names.txt')
            elif self.num_category == 14:
                self.catfile = os.path.join(self.root, 'mymodelnet14_shape_names.txt')
            else:
                self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.DSR:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'Delivery_Service_Robot-train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'Delivery_Service_Robot-test.txt'))]
        else:
            if self.num_category == 8:
                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'mymodelnet8_train.txt'))]
                shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'mymodelnet8_test.txt'))]
            elif self.num_category == 10:
                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'mymodelnet10_train.txt'))]
                shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'mymodelnet10_test.txt'))]
            elif self.num_category == 14:
                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'mymodelnet14_train.txt'))]
                shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'mymodelnet14_test.txt'))]
            else:
                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
                shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]


        assert (split == 'train' or split == 'test')
        if self.DSR:
            shape_names = ['-'.join(x.split('-')[0:-1]) for x in shape_ids[split]]
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.bin') for i in range(len(shape_ids[split]))]
        else:
            shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.fromfile(fn[1], dtype=np.float32).reshape(-1, 4)

        point_set[:, 0:3] = translation_xyz(point_set[:, 0:3])
        point_set[:, 0:3] = normalization(point_set[:, 0:3])

        data_dict = {
            'pts': point_set,
            'target': label
        }
        return data_dict

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/home/PointPillars/Dataset/ModelNet/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
