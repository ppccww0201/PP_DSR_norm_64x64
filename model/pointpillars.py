import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from Voxelization_python import VoxelGenerator


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = VoxelGenerator(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts, batched_pts_ori):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        device = batched_pts[0].device
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out_np, coors_out_np, num_points_per_voxel_out_np = self.voxel_layer.generate(pts.cpu().numpy())
            if len(coors_out_np) == 0:
                voxels_out_np, coors_out_np, num_points_per_voxel_out_np = self.voxel_layer.generate(batched_pts_ori[i].cpu().numpy())
            voxels_out, coors_out, num_points_per_voxel_out = torch.tensor(voxels_out_np).to(device), torch.tensor(coors_out_np[:,[2,1,0]]).to(device), torch.tensor(num_points_per_voxel_out_np).to(device)
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)  # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)  # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        # self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        # self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)


    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # # 1. calculate offset to the points center (in each pillar)
        # offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:,
        #                                                                                            None,
        #                                                                                            None]  # (p1 + p2 + ... + pb, num_points, 3)
        #
        # # 2. calculate offset to the pillar center
        # x_offset_pi_center = pillars[:, :, :1] - (
        #             coors_batch[:, None, 1:2] * self.vx + self.x_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        # y_offset_pi_center = pillars[:, :, 1:2] - (
        #             coors_batch[:, None, 2:3] * self.vy + self.y_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        #
        # # 3. encoder
        # features_ori = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center],
        #                      dim=-1)  # (p1 + p2 + ... + pb, num_points, 9)
        # # In consitent with mmdet3d.
        # # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150
        #
        # # 4. find mask for (0, 0, 0) and update the encoded features
        # # a very beautiful implementation
        # voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        # mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        # mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        # features_ori *= mask[:, :, None]

        # 5. embedding
        points_mean = pillars[:, :, :3].sum(dim=1, keepdim=True) / npoints_per_pillar.type_as(pillars).view(-1, 1, 1)
        f_cluster = pillars[:, :, :3] - points_mean

        f_center = torch.zeros_like(pillars[:, :, :2])
        center = torch.zeros_like(pillars[:, 1, :2]).view(pillars.size()[0], 1, 2)
        center[:, :, 0] = (coors_batch[:, 1].to(pillars.dtype).unsqueeze(1) * self.vx + self.x_offset)
        center[:, :, 1] = (coors_batch[:, 2].to(pillars.dtype).unsqueeze(1) * self.vy + self.y_offset)

        f_center[:, :, 0] = pillars[:, :, 0] - center[:, :, 0]
        f_center[:, :, 1] = pillars[:, :, 1] - center[:, :, 1]

        features = [pillars, f_cluster, f_center]
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(npoints_per_pillar, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(pillars)
        features *= mask

        x = self.linear(features)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        pooling_features = torch.max(x, dim=1, keepdim=True)[0]
        pooling_features = pooling_features.squeeze()

        # features = features.permute(0, 2, 1).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        # features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        # pooling_features_ori = torch.max(features, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1 # bs : batch size
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            temp1 = cur_coors[:,1]
            temp2 = cur_coors[:,2]
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)

        # temp = batched_canvas.detach().cpu().numpy()
        # Encoded Image : batched_canvas
        return batched_canvas

class LeNet5(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2d1 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.relu2d1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.relu2d2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.relu2d3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn2d4 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        # self.relu2d4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=3,stride=2)
        # self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn2d5 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        # self.relu2d5 = nn.ReLU()
        # self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn2d6 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        # self.relu2d6 = nn.ReLU()
        # self.pool6 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc1 = nn.Linear(3136, 128)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-3, momentum=0.01)
        self.relu1d1 = nn.ReLU()
        self.fc2 = nn.Linear(128, nclasses)
        self.bn1d2 = nn.BatchNorm1d(nclasses, eps=1e-3, momentum=0.01)
        self.relu1d2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn2d1(y)
        y = self.relu2d1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2d2(y)
        y = self.relu2d2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.bn2d3(y)
        y = self.relu2d3(y)
        y = self.pool3(y)
        # y = self.conv4(y)
        # y = self.bn2d4(y)
        # y = self.relu2d4(y)
        # y = self.pool4(y)
        # y = self.conv5(y)
        # y = self.bn2d5(y)
        # y = self.relu2d5(y)
        # y = self.pool5(y)
        # y = self.conv6(y)
        # y = self.bn2d6(y)
        # y = self.relu2d6(y)
        # y = self.pool6(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.bn1d1(y)
        y = self.relu1d1(y)
        y = self.fc2(y)
        y = self.bn1d2(y)
        y = self.relu1d2(y)
        y = F.log_softmax(y, dim=1)
        return y

class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=10,
                 voxel_size=[0.082, 0.082, 4],
                 point_cloud_range=[-1, -1, -1, 1, 1, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.pillar_layer = PillarLayer(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=9,
                                            out_channel=64)
        self.LeNet5 = LeNet5(nclasses)

    def forward(self, batched_pts, batched_pts_ori, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts, batched_pts_ori)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        # x: (bs, 384, 248, 216)
        x = self.LeNet5(pillar_features)

        return x, pillar_features

class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        total_loss = loss
        return total_loss
