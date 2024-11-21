#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Chuangji Meng 2021-5-13
import os
import numpy as np
import h5py as h5
# from optionsmcj import set_opts
import options.options as option
import argparse
from optionsmcj import set_opts


args = set_opts()

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='/home/shendi_mcj/code/InvDN-mcj/codes/options/train/train_InvDN-mcj.yml',help='Path to option YMAL file.')
args1 = parser.parse_args()
opt = option.parse(args1.opt, is_train=True)

opt['prepare']['dataroot_Noisy']= '/home/shendi_mcj/datasets/seismic/fielddata/train/original'
opt['prepare']['dataroot_GT']='/home/shendi_mcj/datasets/seismic/fielddata/train/clean'
ori_data_dir=opt['prepare']['dataroot_Noisy']
cle_data_dir=opt['prepare']['dataroot_GT']
print('=> Generating patch samples')

opt['datasets']['train']['dataroot_h5']='/home/shendi_mcj/datasets/seismic/hdf5'
path_h5=opt['datasets']['train']['dataroot_h5'] #train or val
path_h5 = os.path.join(path_h5, 'XJ_trn_s80_90_Patches_256.hdf5') # XJ_Trn_p80_119_Patches_500 XJ_Tst_p120_Patches_500.hdf5 PankeSR_Tst_Patches_128_stride64.hdf5
opt['prepare']['patch_size']=[256, 256]
opt['prepare']['stride']=[128, 128]
patch_size = opt['prepare']['patch_size'] #[128, 128]
stride = opt['prepare']['stride'] #[64, 64]
# train_data_num=opt['prepare']['train_data_num']

from data.prepare_data.segy.big2small_segy import generate_patch_from_segy,generate_patch_from_poststack_segy_1by1
# ori_data_list=generate_patch_from_segy(dir=ori_data_dir,pch_size=patch_size,stride=stride)
# cle_data_list=generate_patch_from_segy(dir=cle_data_dir,pch_size=patch_size,stride=stride)
ori_data_list=generate_patch_from_poststack_segy_1by1(dir=ori_data_dir, pch_size=(256,256),stride=(128,128),jump=1,agc=False,train_data_num=100000,trace_num=20000,section_num=40,aug_times=[],scales = [])
cle_data_list=generate_patch_from_poststack_segy_1by1(dir=cle_data_dir, pch_size=(256,256),stride=(128,128),jump=1,agc=False,train_data_num=100000,trace_num=20000,section_num=40,aug_times=[],scales = [])

ori_max=abs(ori_data_list).max()


# from data.prepare_data.segy.big2small_segy import generate_patch_from_segy_1by1
# ori_data_list = generate_patch_from_segy_1by1(dir=ori_data_dir,pch_size=(256,256),stride=(128, 128),jump=1,agc=False,train_data_num=100000,aug_times=[],scales = [])
# cle_data_list = generate_patch_from_segy_1by1(dir=cle_data_dir,pch_size=(256,256),stride=(128, 128),jump=1,agc=False,train_data_num=100000,aug_times=[],scales = [])
# ori_max=ori_data_list.max()


num_patch = 0
with h5.File(path_h5, 'w') as h5_file:
    for ii in range(len(ori_data_list)):
        if (ii+1) % 10 == 0:
            print('    The {:d} original images'.format(ii+1))

        pch_noisy = ori_data_list[ii]/ori_max
        pch_gt = cle_data_list[ii]/ori_max
        pch_imgs = np.concatenate((pch_noisy, pch_gt), axis=2)
        h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
                               dtype=pch_imgs.dtype, data=pch_imgs)
        num_patch += 1
print('Total {:d} small images in training set'.format(num_patch))
print('Finish!\n')