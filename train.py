import os
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import datetime
import logging
import provider
import argparse
import torchvision
from copy import deepcopy

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader, get_dataloader
from model import PointPillars, get_loss
from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model', default='pointpillar_cls', help='model name')
    parser.add_argument('--num_category', default=8, type=int, choices=[5, 8, 10, 14, 40], help='training on ModelNet10/40')
    parser.add_argument('--Delivery_Service_Robot_Dataset', default=True, type=bool, choices=[True, False],help='training on Delivery_Service_Robot_Dataset')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def test(model, loader, num_class=10):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

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

        pred, pseudo_image = classifier(batched_pts, batched_pts)
        pred_choice = pred.data.max(1)[1]

        # temp = torch.reshape(pseudo_image, (64, 1, 24, 24))
        # x = torchvision.utils.make_grid(temp, normalize=True)
        # writer.add_image('pseduo_image/train', x, j)

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
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../Delivery_Service_Robot_Dataset' if args.Delivery_Service_Robot_Dataset else '../modelnet40_normal_resampled/'
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    trainDataLoader = get_dataloader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True)
    testDataLoader = get_dataloader(dataset=test_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True)

    '''MODEL LOADING'''
    num_class = args.num_category
    PC_range = [-1, -1, -1, 1, 1, 1]
    classifier = PointPillars(nclasses=num_class,
                 voxel_size=[element * 1 for element in [0.03125, 0.03125, 0.2]],
                 point_cloud_range=[element * 1 for element in PC_range],
                 # voxel_size=[element * 1 for element in [0.104, 0.166, 2.5]],
                 # point_cloud_range=[element * 1 for element in [-1.5, 1, -1, 1, 5, 1.5]],
                 # voxel_size=[element * 1 for element in [0.125, 0.175, 2.4]],
                 # point_cloud_range=[element * 1 for element in [-1.5, -2.1, -1.2, 1.5, 2.1, 1.2]],
                 max_num_points=32,
                 max_voxels=30000)
    criterion = get_loss()

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    try:
        checkpoint = torch.load('log/classification/2023-11-24_01-13/checkpoints/best_model.pth')
        # checkpoint = torch.load('log/classification/2023-05-17_09-22' + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        # for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        for i, data_dict in enumerate(tqdm(trainDataLoader)):
            batched_pts, batched_target = data_dict['batched_pts'], data_dict['batched_labels']
            batched_pts_ori = deepcopy(batched_pts)
            optimizer.zero_grad()
            batch_size = len(batched_pts)

            for batch_idx in range(batch_size):
                points = batched_pts[batch_idx]
                points = points.data.numpy()
                points = provider.random_point_dropout(points[np.newaxis, :, :])
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], shift_range=0.1)
                points = torch.Tensor(points).squeeze()
                batched_pts[batch_idx] = points if args.use_cpu else points.cuda()

            target = torch.stack(batched_target, axis=0).squeeze()
            if not args.use_cpu:
                target = target.cuda()

            pred, pseudo_image = classifier(batched_pts, batched_pts_ori)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(batch_size))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    writer = SummaryWriter()
    main(args)