#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import mcubes
# import marching_cubes as mcubes
import logging
from tqdm import tqdm, trange
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, create_nerf
from nerf_sample_ray_split import CameraManager
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__package__)

def ddp_mesh_nerf(rank, args):
    ###### set up multi-processing
    assert(args.world_size==1)
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    camera_mgr = CameraManager(learnable=False)
    start, models = create_nerf(rank, args, camera_mgr, False)

    # center on lk
    current_min = 0
    current_max = 300
    current_min_array = np.array([current_min, current_min, current_min])
    current_max_array = np.array([current_max, current_max, current_max])
    ax = np.linspace(-1, 1, num=current_max, endpoint=True, dtype=np.float32)
    # X, Y, Z = np.meshgrid(ax, ax, ax+0.4)
    X, Y, Z = np.meshgrid(ax, ax, ax)

    # flip yz
    pts = np.stack((X, Y[::-1], Z[::-1]), -1)/4
    pts = pts.reshape((-1, 3))

    pts = torch.tensor(pts).float().to(rank)

    target_min = pts.min().cpu().numpy()
    target_max = pts.max().cpu().numpy()
    target_min_array = np.array([target_min, target_min, target_min])
    target_max_array = np.array([target_max, target_max, target_max])

    u = models['net_1']
    nerf_net = u.module.nerf_net
    fg_net = nerf_net.fg_net

    allres = []
    allcolor = []
    with autocast():
        with torch.no_grad():
            # direction = torch.tensor([0, 0, -1], dtype=torch.float32).to(rank)
            for bid in trange((pts.shape[0]+args.chunk_size-1)//args.chunk_size):
                bstart = bid * args.chunk_size
                bend = bstart + args.chunk_size
                cpts = pts[bstart:bend]
                cvd = cpts*0#+direction

                out = fg_net(cpts, cvd, iteration=start,
                             embedder_position=nerf_net.fg_embedder_position,
                             embedder_viewdir=nerf_net.fg_embedder_viewdir)

                res = out['sigma'].detach().cpu().numpy()
                allres.append(res)
                color = out['rgb'].detach().cpu().numpy()
                allcolor.append(color)

    allres = np.concatenate(allres, 0)
    allres = allres.reshape(X.shape)

    allcolor = np.concatenate(allcolor, 0)
    allcolor = allcolor.reshape(list(X.shape)+[3,])

    # print(allres.min(), allres.max(), allres.mean(), np.median(allres), allres.shape)

    def create_point_cloud(allres, allcolor, current_min, current_max, target_min, target_max, threshold=0.1):
        # 确定有效点的索引
        valid_points = allres > threshold

        # 获取点的坐标、颜色和法线（这里法线为0）
        points = np.argwhere(valid_points)

        # 计算缩放因子, 应用缩放和偏移
        scale = (target_max - target_min) / (current_max - current_min)
        points = (points - current_min) * scale + target_min

        colors = allcolor[valid_points] * 255
        normals = np.zeros_like(points, dtype=np.float32)

        # 将坐标、颜色和法线合并
        point_cloud = np.hstack((points, colors, normals))

        return point_cloud

    def save_point_cloud(point_cloud, filename):
        # 创建PLY元素，包含法线信息
        vertex = np.array([tuple(p) for p in point_cloud],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')

        # 写入PLY文件
        PlyData([el]).write(filename)

    # 使用函数
    config_name = args.config
    seq = config_name.split('/')[-2] + '_' + config_name.split('/')[-1].split('.')[0]
    point_cloud = create_point_cloud(allres, allcolor, current_min_array, current_max_array, target_min_array, target_max_array, threshold=0.1)
    save_point_cloud(point_cloud, (args.basedir + '/' + seq + '/' + 'points3D.ply'))

    # def create_point_cloud(allres, allcolor, threshold=0.1):
    #     # 确定有效点的索引
    #     valid_points = allres > threshold
    # 
    #     # 获取点的坐标和颜色
    #     points = np.argwhere(valid_points)
    #     colors = allcolor[valid_points]
    # 
    #     # 将坐标和颜色合并
    #     point_cloud = np.hstack((points, colors))
    # 
    #     return point_cloud
    # 
    # def save_point_cloud(point_cloud, filename):
    #     # 创建PLY元素
    #     vertex = np.array([tuple(p) for p in point_cloud],
    #                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    #     el = PlyElement.describe(vertex, 'vertex')
    # 
    #     # 写入PLY文件
    #     PlyData([el]).write(filename)
    # 
    # # 转换和保存点云
    # config_name = args.config
    # seq = config_name.split('/')[-2] + '_' + config_name.split('/')[-1].split('.')[0]
    # point_cloud = create_point_cloud(allres, allcolor, threshold=0.1)
    # save_point_cloud(point_cloud, (args.basedir + '/' + seq + '/' + 'output.ply'))

    # # 保存allres和allcolor
    # config_name = args.config
    # seq = config_name.split('/')[-2] + '_' + config_name.split('/')[-1].split('.')[0]
    # np.save((args.basedir + '/' + seq + '/' + 'allres.npy'), allres)
    # np.save((args.basedir + '/' + seq + '/' + 'allcolor.npy'), allcolor)

    # logger.info('Doing MC')
    # vtx, tri = mcubes.marching_cubes(allres.astype(np.float32), 100)
    # THR=30
    # # vtx, tri = mcubes.marching_cubes_color(allres.astype(np.float32), allcolor.astype(np.float32), THR)
    # logger.info('Exporting mesh')
    # config_name = args.config
    # seq = config_name.split('/')[-2] + '_' + config_name.split('/')[-1].split('.')[0]
    # mcubes.export_obj(vtx, tri, (args.basedir + '/' + seq + '/' + "mesh5.obj"))
    # # mcubes.export_mesh(vtx, tri, (args.basedir + '/' + seq + '/' + "mesh5.dae"), "Mesh")
    # # mcubes.export_obj(vtx, tri, f"colornet01_scale4_{THR}.obj")


def mesh():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_mesh_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    mesh()

