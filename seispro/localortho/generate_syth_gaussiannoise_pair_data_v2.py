

import numpy as np
import matplotlib.pyplot as plt
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
from seispro.data.prepare_data.mat.bia2small_mat import generate_patch_from_mat,generate_patch_from_mat2
marmousi_im_dir = 'G:\datasets\seismic\marmousi'

# from scipy.io import loadmat,savemat
# marmousi35 = loadmat(marmousi_im_dir+'\marmousi35.mat')['data']
# max=abs(marmousi35).max()
# marmousi35_gn005= marmousi35 +np.random.normal(0,12.75/255*max,marmousi35.shape)
# savemat((marmousi_im_dir+'\marmousi35_gn005.mat'), {'data': marmousi35_gn005})


#  该函数  加了  /*35.mat  /*mp
marmousi_im_list = generate_patch_from_mat(dir=marmousi_im_dir, pch_size=256, stride=[256, 256]).squeeze()
marmousi_noisy_list = generate_patch_from_mat2(dir=marmousi_im_dir, pch_size=256, stride=[256, 256]).squeeze()

ori_max=abs(marmousi_noisy_list).max()
path_h5=marmousi_im_dir
path_h5 = os.path.join(path_h5, 'marmousi35_trn_gaussian005_noise_Patches_256_1.hdf5')#


num_patch = 0
import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
with h5.File(path_h5, 'w') as h5_file:
    for ii in range(len(marmousi_im_list)):
        if (ii+1) % 10 == 0:
            print('    The {:d} original images'.format(ii+1))
        pch_noisy = marmousi_noisy_list[ii].squeeze()/ori_max
        pch_dn = marmousi_im_list[ii].squeeze()/ori_max
        if ii == 0:
            show(pch_noisy,pch_dn,method='GT')
        # noise_data=pch_noisy-pch_dn
        pch_imgs = np.concatenate((np.expand_dims(pch_noisy,axis=2), np.expand_dims(pch_dn,axis=2)), axis=2)
        h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
                               dtype=pch_imgs.dtype, data=pch_imgs)
        num_patch += 1
print('Total {:d} small paired data in training set'.format(num_patch))
print('Finish!\n')
