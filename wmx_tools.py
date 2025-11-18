import os, torch
import numpy as np

from .TransPose import articulate as art
from .TransPose.config import paths, joint_set, acc_scale, vel_scale


def _full_local_mat_to_reduced_glb_6d(pose: torch.Tensor):
    r"""
    利用24个关节的相对姿态, 转化为神经网络输入的全局姿态

    :param pose: Tensor in shape [num_frame, 24, 3, 3] (6d represention of net output)
    """
    m = art.ParametricModel(os.path.join('/content/drive/MyDrive/TransPose',paths.smpl_file))
    global_full_pose = m.forward_kinematics_R(pose).view(-1, 24, 3, 3)
    glb_reduced_pose = global_full_pose[:, joint_set.reduced]
    glb_reduced_pose = art.math.rotation_matrix_to_r6d(glb_reduced_pose).view(-1, joint_set.n_reduced, 6)
    
    return glb_reduced_pose


def generate_acc_noise(glb_acc: torch.Tensor, glb_ori: torch.Tensor):
    # 加速度计噪声包括呼吸造成的加速度噪声与随机游走
    acc_noise = torch.zeros(glb_acc.shape); 
    phi = 2 * np.pi * torch.rand(1)
    breath = 0.2 * torch.sin(torch.linspace(0, 2 * np.pi * (glb_acc.shape[0] / 60 / 60 * 90), glb_acc.shape[0]) + phi).unsqueeze(1) * torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0)
    acc_noise[:, -1, :] = glb_ori[:, -1, :, :].bmm(breath.unsqueeze(-1)).squeeze(-1)

    acc_noise = acc_noise + 0.02 * torch.randn(glb_acc.shape) * torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0) # 0.02 m/s2 的随机游走

    return acc_noise / acc_scale

def generate_gyro_noise(glb_ori: torch.Tensor):
    r"""
    利用腰部IMU的全局姿态与神经网络输入的全局姿态, 转化为24个关节的相对姿态, 忽略的关节变为单位阵

    :param glb_ori: Tensor in shape [num_frame, 3, 3] single imu rotation in globle
    """
    deg2rad = np.pi / 180

    time_s = glb_ori.shape[0] / 60
    ori_bias = torch.zeros(glb_ori.shape[0], 3)
    ori_bias[:, 0] = torch.rand(glb_ori.shape[0]) * torch.linspace(0, time_s * 1 * deg2rad / 3600, glb_ori.shape[0]) # 1deg/h 的航向零偏(俯仰横滚没有)
    ori_random = 5 * deg2rad * torch.randn(glb_ori.shape[0], 3)  # 随机噪声1deg(均匀分布)
    ori_noise = ori_bias + ori_random

    return art.math.angular.euler_angle_to_rotation_matrix(ori_noise)


def normalize_and_concat_X(glb_acc: torch.Tensor, glb_ori: torch.Tensor, Add_Noise = True):
    glb_acc = glb_acc.view(-1, 3, 3)
    glb_ori = glb_ori.view(-1, 3, 3, 3)

    if (Add_Noise):
        for i in range(3):
            glb_ori[:, i, :, :] = generate_gyro_noise(glb_ori[:, i, :, :]).bmm(glb_ori[:, i, :, :])
        # 添加噪声
        glb_acc = glb_acc + generate_acc_noise(glb_acc, glb_ori)

    acc = torch.cat(
        (glb_acc[:, :2] - glb_acc[:, 2:], glb_acc[:, 2:]), dim=1
        ).bmm(glb_ori[:, -1]) / acc_scale
    ori = torch.cat(
        (glb_ori[:, 2:].transpose(2, 3).matmul(glb_ori[:, :2]),
         glb_ori[:, 2:])
         , dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data


def normalize_and_concat_y(glb_tran: torch.Tensor, local_pose: torch.Tensor, local_joint: torch.Tensor, glb_ori: torch.Tensor, shape: torch.Tensor):
    root_ori = glb_ori.view(-1, 3, 3, 3)[:, -1, :, :]
    local_full_joint_position = (local_joint[:, joint_set.full, :] - glb_tran.unsqueeze(1)).bmm(root_ori)

    m = art.ParametricModel(paths.smpl_file)
    j, _ = m.get_zero_pose_joint_and_vertex(shape)
    b = art.math.joint_position_to_bone_vector(j.squeeze(0)[joint_set.full].unsqueeze(0),
                                            #[None,  0, 0, 1, 2, 4, 5, 7, 8]
                                            joint_set.full_parent
                                            ).squeeze(0)
    bone_length = art.math.normalize_tensor(b, return_norm=True)[1][1:9].squeeze(1).repeat(local_full_joint_position.shape[0], 1)

    velocity = glb_tran.diff(1, dim = 0, append=torch.zeros(1, glb_tran.shape[1])).unsqueeze(1).bmm(root_ori).squeeze(1) * 60 / vel_scale

    contact_probability = torch.cat([
        (local_joint[:, -2, :].diff(1, dim = 0, append=torch.zeros(1, local_joint[:, -2, :].shape[1])).norm(dim = 1) > 0.01).float().unsqueeze(dim = 1),
        (local_joint[:, -1, :].diff(1, dim = 0, append=torch.zeros(1, local_joint[:, -1, :].shape[1])).norm(dim = 1) > 0.01).float().unsqueeze(dim = 1)
    ], dim = 1)
#局部坐标系中关节的完整位置信息  骨骼的长度信息  两个特定关节的接触概率  全局平移的速度信息
    return local_full_joint_position, bone_length, contact_probability, velocity    


# 参数配置------------------------------------------------------------------------

import argparse, yaml

def create_parser(bMain = False):
    parser = argparse.ArgumentParser()
    if not bMain:
        parser.add_argument('--mode_params',    type = float, nargs='+', help = 'params for multi-stage training')

        parser.add_argument('--learning_rate',  type = float, default = 1e-5, help = 'learning_rate')
        parser.add_argument('--weight_decay',   type = float, default = 0.0, help = 'weight decay for optimizer')

        parser.add_argument('--epoch',          type = int, default = 100,  help = 'train epoch number')
        parser.add_argument('--early_stopping', type = float, default = 10, help = 'early stoopping epoch')

        parser.add_argument('--use_amp',        action='store_true', help ='if use auto amp in neural network')

    else:
        parser.add_argument('--window_size',        type = int, default = 300,  help = 'signal window size')
        parser.add_argument('--window_step',        type = int, default = 1,  help = 'signal window size step')
        parser.add_argument('--batch_size',     type = int,   default = 8, help = 'batch size')
        parser.add_argument('--batch_size_eval',type = int,   default = 128, help = 'batch size')
        parser.add_argument('--test_size',      type = float, default = 0.4, help = 'test size of data size')

    return parser

def create_args(yaml_path):
    parser = create_parser()
    # -------------------------------------------------------------------------------------------
    args1 = parser.parse_args(args=[          
        '--mode_params',            '1.0', '0.0', '0.0',

        '--learning_rate',          '1e-5',
        '--weight_decay',           '1e-3',
        '--epoch',                  '500',
        '--early_stopping',         '30',

        #'--use_amp', 
    ])
    # print(args1)

    args2 = parser.parse_args(args=[
        #'--create_data',           
        '--mode_params',            '0.0', '0.0', '1.0',

        '--learning_rate',          '1e-3',
        '--weight_decay',           '1e-3',
        '--epoch',                  '500',
        '--early_stopping',         '30',

        #'--use_amp', 
    ])
    # print(args2)

    args3 = parser.parse_args(args=[
        #'--create_data',           
        '--mode_params',            '0.0', '1.0', '0.0',

        '--learning_rate',          '1e-3',
        '--weight_decay',           '1e-3',
        '--epoch',                  '500',
        '--early_stopping',         '30',

        #'--use_amp', 
    ])

    args_all = create_parser(True).parse_args(args=[
        '--batch_size',             '256', #'32',
        '--batch_size_eval',         '2048', #'64',

        '--window_size',            '100',
        '--window_step',            '20',
    ])

    with open(yaml_path, "w", encoding = "utf-8") as f:
        temp_write = args_all.__dict__

        # args1.weight_path = f"./weight/model_{time_path}_s1.pt"
        temp_write['stage1'] = args1.__dict__

        # args2.weight_path = f"./weight/model_{time_path}_s2.pt"
        temp_write['stage2'] = args2.__dict__

        # args3.weight_path = f"./weight/model_{time_path}_s3.pt"
        temp_write['stage3'] = args3.__dict__

        yaml.dump(temp_write, f)