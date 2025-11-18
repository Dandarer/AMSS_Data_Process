r"""
    Config for paths, joint set, and normalizing scales.
"""

import os
from pathlib import Path

# datasets (directory names) in AMASS
# e.g., for ACCAD, the path should be `paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`

amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 
              'SSM_synced', 'CMU', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 
              'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD', 'BioMotionLab_NTroje', 
              'BMLhandball', 'MPI_Limits', 'DFaust67']

# amass_data = ['HumanEva']
# amass_data = ['HumanEva', 'MPI_HDM05', 'Transitions_mocap']

UTILS_PATH = str(Path(os.path.abspath(__file__)).parents[0].parents[0])
ROOT_PATH = str(Path(os.path.abspath(__file__)).parents[0].parents[0].parents[0])

class paths:
    raw_amass_dir = "/mnt/data/dataset/AMASS_DATASET/AMASS"             # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = os.path.join(ROOT_PATH, 'data/dataset_work/AMASS')         # output path for the synthetic AMASS dataset

    raw_dipimu_dir = "/mnt/data/dataset/DIP_DATASET/DIP_IMU"   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = os.path.join(ROOT_PATH, 'data/dataset_work/DIP_IMU')      # output path for the preprocessed DIP-IMU dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dip_dir = "FALSE"  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = os.path.join(UTILS_PATH, 'TransPose/data/dataset_raw/TotalCapture/official')    # contain official gt (S1/acting1/gt_skel_gbl_pos.txt)
    totalcapture_dir = os.path.join(ROOT_PATH, 'data/dataset_work/TotalCapture')          # output path for the preprocessed TotalCapture dataset

    example_dir = os.path.join(UTILS_PATH, 'TransPose/data/example')                    # example IMU measurements
    smpl_file = os.path.join(UTILS_PATH, 'TransPose/models/SMPL_male.pkl')              # official SMPL model path
    weights_file = os.path.join(UTILS_PATH, 'TransPose/data/weights.pt')                # network weight file


class joint_set:
    # 3IMU模式
    leaf = [7, 8] # IMU位置，还有一个0
    full =          [1, 2, 4, 5, 7, 8] # 位置
    full_parent =   [0, 0, 1, 2, 3, 4]
    reduced = [1, 2, 4, 5]  # 关节
    ignored = [0, 3, 6, 7, 8, 10, 11, 20, 21, 22, 23, 9, 12, 13, 14, 15, 16, 17, 18, 19]

    # 6IMU模式
    # leaf = [7, 8, 12, 20, 21] # IMU位置，还有一个0
    # full = list(range(1, 24))
    # reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    # ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    # 下半身
    lower_body =        [   0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4,  5,  6] # 是lower_body的父节点, 序号表示lower_body列表的位置，一一对应

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


acc_scale = 30
vel_scale = 3
