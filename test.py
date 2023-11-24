from data_utils.ModelNetDataLoader import ModelNetDataLoader, get_dataloader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from model import PointPillars
from tensorboardX import SummaryWriter
import torchvision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_category', default=5, type=int, choices=[5,8,10,14,40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
    for i, data_dict in enumerate(tqdm(loader)):
        batched_pts, batched_target = data_dict['batched_pts'], data_dict['batched_labels']
        batch_size = len(batched_pts)

        for batch_idx in range(batch_size):
            points = batched_pts[batch_idx].squeeze()
            batched_pts[batch_idx] = points if args.use_cpu else points.cuda()

        target = torch.stack(batched_target, axis=0).squeeze()
        if not args.use_cpu:
            target = target.cuda()

        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, pseudo_image = classifier(batched_pts)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        # Image Save
        #temp = torch.reshape(pseudo_image, (64,1,24,24))
        #x = torchvision.utils.make_grid(temp, normalize=True)
        #writer.add_image('pseduo_image/test', x, j)

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float((target == cat).sum())
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(batch_size))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''DATA LOADING'''
    data_path = '/home/PointPillars/Dataset/ModelNet/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = get_dataloader(dataset=test_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True)

    '''MODEL LOADING'''
    num_class = args.num_category
    classifier = PointPillars(nclasses=num_class,
                 voxel_size=[element * 1 for element in [0.082, 0.082, 2]],
                 point_cloud_range=[element * 1 for element in [-1, -1, -1, 1, 1, 1]],
                 max_num_points=32,
                 max_voxels=30000)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load('log/classification/2023-11-17_02-02/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

if __name__ == '__main__':
    args = parse_args()
    writer = SummaryWriter()
    main(args)