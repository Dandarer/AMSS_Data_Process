"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""


from . import articulate as art
import torch
import os
import pickle
from .config import paths, amass_data
import numpy as np
from tqdm.notebook import tqdm
import glob

def process_amass(smooth_n=4):
    """
    预处理AMASS数据，生成所需的伪IMU数据
    params: smooth_n 代表平滑窗口，与前后数据共n个平滑
    """

    # # vertice 来源于绘制vert散点图，一一对应
    # 6IMU模式
    # vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    # # joint index来源于SMPL论文 
    # # https://zhuanlan.zhihu.com/p/256358005
    # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) 

    print("shape.pt: SMPL parameter(没什么关系)")
    print(" pose.pt: 24个关节姿态的轴角表示")
    print("joint.pt: 24个关节相对位置")
    print(" tran.pt: 人物动作的绝对位置(与绝对位置结合可得真实位置)")
    print(" vrot.pt: 生成的陀螺数据(右脚尖，左脚尖，后腰)，实际上是姿态旋转矩阵，需要AHRS算出姿态")
    print(" vacc.pt: 生成的n系加速度计数据(右脚尖，左脚尖，后腰)，需要根据姿态旋转到n系东北天")
    print("有用的关节：     11,     10,     8,     7,      5,     4,       2,       1")
    print("    对应为： 右脚尖， 左脚尖；右脚踝，左脚踝；右膝盖，左膝盖；右大腿根，左大腿根")
    print("#"*50)

    # 3IMU模式
    vi_mask = torch.tensor([1176, 4662, 3021,
                            ]) # 右脚踝，左脚踝, 后腰
    ji_mask = torch.tensor([4, 5, 0, 
                        ]) #右脚踝，左脚踝；

   #将位置积分为加速度的函数
    def _syn_acc(v):
        """
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    body_model = art.ParametricModel(paths.smpl_file)




    #该代码是AMASS数据集的标准预处理流程，主要完成：
    #多数据集合并
    #帧率标准化（60fps）
    #数据格式转换（numpy数组存储）
    #参数维度精简（β参数截断）

    #初始化四个列表  分别为人体姿态参数pose，人体位置参数tran，人体形状参数beta，每个动作序列的帧数
    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])





    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)   #调整张量的形状  -1表示自动计算该维度的大小  52表示姿态数据的关节数量和每个关节的维度（三维姿态）
    
    #pose维度为3，下述只对前两个维度做处理                                                                         #形状则为（M,52,3），M是姿态数据的帧数
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body   #保留24个关节的姿态数据，即身体主干  不含手指

    # # align AMASS global fame with DIP#(左前下)
    # amass_rot = torch.tensor([[ [1, 0, 0], 
    #                             [0, 0, 1], 
    #                             [0, -1, 0.]]])
    # tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    # pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
    #     amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    #初始化多个空列表 分别为姿态参数(轴角表示)、形状参数、全局平移参数、关节绝对位置、关节旋转矩阵(陀螺数据)、顶点加速度（加速度计数据）
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        #将pose数据转换为旋转矩阵（序列长度，24个关节，3×3旋转矩阵）
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        #调用前向运动学模型，分别计算全局旋转矩阵、关节的3D位置、网格顶点坐标
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

        out_pose.append(pose[b:b + l].clone())  # N, 24, 3         24个关节的轴角表示
        out_tran.append(tran[b:b + l].clone())  # N, 3             保存全局平移参数（人物在空间中的绝对位置）
        out_shape.append(shape[i].clone())  # 10                   保存SMPL形状参数（描述人体体型）
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3           保存24个关节的绝对3D位置
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 3, 3     第一个N为帧数  即时间序列的长度  第一个3为顶点的网格坐标（三个顶点）  第二个3为加速度分量xyz
        out_vrot.append(grot[:, ji_mask])  # N, 3, 3, 3            第一个N为帧数  即时间序列的长度  第一个2为关节的数量（两个关节） 后面两个3为旋转矩阵
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))      # 24个关节的轴角表示
    torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))    # SMPL parameter(没什么关系)
    torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))      # 人物动作的绝对位置
    torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))    # 24个关节绝对位置
    torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))      # 生成的陀螺数据(实际上是姿态旋转矩阵，需要AHRS算出姿态)
    torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))      # 生成的加速度计数据
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)


def process_dipimu():
    imu_mask = [11, 12, 2]
    #imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)


def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)


if __name__ == '__main__':
    process_amass()
    # process_dipimu()
    # process_totalcapture()
