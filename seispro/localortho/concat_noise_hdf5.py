import h5py as h5
import numpy as np

def show(x,method):
    import matplotlib.pyplot as plt
    noise= x.squeeze()
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    # io.savemat(('../../noise/shot_' + method + '_n.mat'), {'data': noise[:, :, np.newaxis]})
    # plt.title('removed noise')
    plt.tight_layout()
    plt.show()

XJ_hdf5_path='E:\博士期间资料\田亚军\\2021.6.07\original\XJ_trn_s110_120_Patches_256_orthofxdecon.hdf5'
PK_hdf5_path='E:\博士期间资料\田亚军\\2021.6.07\original\PK_trn_Patches_256_orthofxdecon.hdf5'
hdf5_path_list=[XJ_hdf5_path,PK_hdf5_path]
concat_hdf5_path='E:\博士期间资料\田亚军\\2021.6.07\original\XJ_PK_noise_Patches_256_orthofxdecon.hdf5'
noise_list=[]
num_patch = 0
for hdf_path in hdf5_path_list:
    with h5.File(hdf_path, 'r') as h5_file:
        keys = list(h5_file.keys())
        num_images = len(keys)
        for ii in range(len(keys)):
            if (ii + 1) % 10 == 0:
                print('    The {:d} original images'.format(ii + 1))
            imgs_sets = h5_file[keys[ii]]
            H, W, C2 = imgs_sets.shape
            # minus the bayer patter channel
            C = int(C2 / 2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
            noise=im_noisy-im_gt
            if ii == 0:
                show(noise, method='orthofxdecon')
            num_patch += 1
            noise_list.append(noise)
    print('Single Dataset Finish!\n')

num_patch_c=0
with h5.File(concat_hdf5_path, 'w') as h5_file:
    for jj in range(len(noise_list)):
        if (jj + 1) % 10 == 0:
            print('The {:d} noise images'.format(jj + 1))
        noise = noise_list[jj]
        h5_file.create_dataset(name=str(num_patch_c), shape=noise.shape,
                                   dtype=noise.dtype, data=noise)
        num_patch_c += 1
print('Total {:d} small noise data in noise dataset'.format(num_patch_c))
print('Finish!\n')