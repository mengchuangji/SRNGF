import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import math
from seis_util.gain import gain

def compare_SNR(real_img,recov_img):
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var ==0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var,10)
    return s
import scipy.io as io

def show_NyDnNo_single(y,x_,method,dpi=300,figsize=(10, 6),clip=0.2):

    import matplotlib.pyplot as plt
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 12,
    #          }
    method=method
    fontsize=24
    labelfontsize=20
    clip =abs(y).max()*clip #1 #abs(x).max()  # 显示范围，负值越大越明显  用x好一些
    vmin, vmax = -clip, clip
    fig=plt.figure(dpi=dpi, figsize=figsize)  # 16,3 (26, 3)dpi=500 (18,5)


    plt.rcParams["font.family"] = "Times New Roman"
    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2
    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap='gray') #'gray' plt.cm.seismic
    # plt.title('Noisy section', fontsize=fontsize, y=-0.1)

    fig = plt.figure(dpi=dpi, figsize=figsize)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax,cmap='gray')#,
    # plt.title('Denoised section',fontsize=fontsize,y=-0.1)

    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/salt_'+method+'_dn.mat'), {'data': x_[:, :, np.newaxis]})
    #############

    fig = plt.figure(dpi=dpi, figsize=figsize)
    noise = y - x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap='gray')
    # plt.title('Removed noise section', fontsize=fontsize, y=-0.1)
    # plt.colorbar(shrink=0.5)
    # io.savemat(('../../noise/salt_'+method+'_n.mat'), {'data': noise[:, :, np.newaxis]})
    #../../

    # plt.colorbar(shrink=0.8)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0,hspace=0)

    # plt.savefig('E:\VIRI\paper\\1stPaperSE\\revise\output\method_compare_output\\'+method+'.png', format='png', dpi=500,
    #             bbox_inches='tight')
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})

