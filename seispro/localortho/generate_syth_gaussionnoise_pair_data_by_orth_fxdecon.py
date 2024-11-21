
# from seispro.psnr import psnr
import numpy as np
import matplotlib.pyplot as plt
import segyio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matlab.engine
import matlab
import math
import os
import scipy.io as io
import glob
import h5py as h5

def show(x,y,method):
    import matplotlib.pyplot as plt
    plt.figure(dpi=500,figsize=(6,10)) #(12,9)
    plt.subplot(131)
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray') #'gray' #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(132)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    # plt.title('denoised')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/shot_' + method + '_dn.mat'), {'data': y[:, :, np.newaxis]})

    plt.subplot(133)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    # io.savemat(('../../noise/shot_' + method + '_n.mat'), {'data': noise[:, :, np.newaxis]})
    # plt.title('removed noise')
    plt.tight_layout()
    plt.show()

# 制作marmousi+field noise 的训练集
from seispro.data.prepare_data.mat.bia2small_mat import generate_patch_from_mat,generate_patch_from_noisy_mat
marmousi_im_dir = 'G:\datasets\seismic\marmousi'
#  该函数  加了  /*35.mat  /*mp
marmousi_im_list = generate_patch_from_noisy_mat(dir=marmousi_im_dir, pch_size=256, stride=[256, 256],sigma=12.75).squeeze()

path_h5=marmousi_im_dir
path_h5 = os.path.join(path_h5, 'marmousi35_trn_gaussian005_noise_Patches_256_orthofxdecon.hdf5')#

# # 制作overthrust+field noise 的测试集
# from seispro.data.prepare_data.mat.bia2small_mat import generate_patch_from_mat
# overthrust_im_dir = 'D:\datasets\seismic\overthrust'  #(1,187,801)
# overthrust_im_list = generate_patch_from_mat(dir=overthrust_im_dir, pch_size=64, stride=[64, 64]).squeeze()
# salt_im_dir = 'D:\datasets\seismic\salt'  #(1,187,801)
# salt_im_list = generate_patch_from_mat(dir=salt_im_dir, pch_size=64, stride=[64, 64]).squeeze()
# train_im_list=np.concatenate([overthrust_im_list, salt_im_list], axis=0)
#
# XJ_segy_dir='E:\博士期间资料\田亚军\\2021.6.07\original'
# from seispro.seis_utils.readsegy import generate_patch_from_poststack_segy_1by1
# #  该函数  加了  *N.s*gy
# XJ_noise_list = generate_patch_from_poststack_segy_1by1(dir=XJ_segy_dir, pch_size=(64,64),stride=(64,64),jump=1,agc=False,train_data_num=100000,trace_num=20000,section_num=40,aug_times=[],scales = [])
# # ori_max=abs(XJ_im_list).max()
# XJ_im_list=train_im_list+XJ_noise_list[0:train_im_list.shape[0],:,:]
# path_h5=overthrust_im_dir
# path_h5 = os.path.join(path_h5, 'overthrust_salt_trn_XJ_s120_noise_Patches_256_orthofxdecon.hdf5')#

num_patch = 0
import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
with h5.File(path_h5, 'w') as h5_file:
    for ii in range(len(marmousi_im_list)):
        if (ii+1) % 10 == 0:
            print('    The {:d} original images'.format(ii+1))
        pch_noisy = marmousi_im_list[ii].squeeze() #/ori_max
        pch_dn1 = eng.fx_decon(matlab.double(pch_noisy.tolist()),matlab.double([0.001]),matlab.double([15]),matlab.double([0.01]),matlab.double([1]),matlab.double([124])); #XJ 0.004 PK 0.002
        noise_data1 = pch_noisy - np.array(pch_dn1)
        pch_dn2, noise_data2, low = eng.localortho(matlab.double(pch_dn1),
                                                         matlab.double(noise_data1.tolist()),
                                                         matlab.double([[20, 20, 1]]),
                                                         100,
                                                         0.0,
                                                         1, nargout=3)  # 20,0,1
        pch_dn = np.array(pch_dn2)
        if ii == 0:
            show(pch_noisy,pch_dn,method='orthofxdecon')
        # noise_data=pch_noisy-pch_dn
        pch_imgs = np.concatenate((np.expand_dims(pch_noisy,axis=2), np.expand_dims(pch_dn,axis=2)), axis=2)
        h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
                               dtype=pch_imgs.dtype, data=pch_imgs)
        num_patch += 1
print('Total {:d} small paired data in training set'.format(num_patch))
print('Finish!\n')
