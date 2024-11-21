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
#x:n
# noisy y:denoised

def show(x,y,x_):
    import matplotlib.pyplot as plt
    clip = abs(y).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    plt.figure(figsize=(16, 30))  # 16,3
    plt.subplot(611)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)  # cmap='gray'
    plt.title('clean data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(612)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('noisy data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(613)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('denoised data')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('./noise/vdn_dn_uns_25.mat'), {'data': x_[:, :, np.newaxis]})

    plt.subplot(614)
    noise = y - x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('removed noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(615)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(616)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_ - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('residual')
    # plt.colorbar(shrink=0.5)
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})
def show_xynss(x,y,sigma2,simi):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,9)) #(12,9)

    plt.subplot(151)
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(152)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(153)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    plt.title('removed noise')


    plt.subplot(154)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(sigma2)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('predicted sigma')
    plt.colorbar(shrink=0.5)
    plt.tight_layout()

    plt.subplot(155)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(simi,vmin=0,vmax=1,cmap='jet')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('local similarity')
    plt.colorbar(shrink=0.5)
    plt.tight_layout()

    plt.show()

def show_xyns(x,y,s):
    import matplotlib.pyplot as plt
    clip=abs(x).max()
    vmin, vmax = -clip, clip
    plt.figure(figsize=(9,10)) #(12,9)
    plt.subplot(141)
    plt.imshow(x,vmin=vmin,vmax=vmax,cmap='gray') #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(142)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=vmin,vmax=vmax,cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(143)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=vmin,vmax=vmax,cmap='gray')
    plt.title('removed noise')

    plt.subplot(144)
    simi= s
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(simi,vmin=0,vmax=1,cmap='jet')
    plt.colorbar(shrink=0.8)
    plt.title('local similarity')
    plt.tight_layout()
    plt.show()

def show_yxnSigma_gain(y,x_,sigma,agc):
    noise = y - x_
    if agc:
        y = gain(y, 0.004, 'agc', 0.05, 1)
        x_ = gain(x_, 0.004, 'agc', 0.05, 1)
        noise = gain(noise, 0.004, 'agc', 0.05, 1)

    clip = abs(y).max()
    vmin, vmax = -clip, clip
    import matplotlib.pyplot as plt
    fontsize=4
    plt.figure(dpi=500,figsize=(8,10)) #(12,9) (8,10)
    plt.subplot(141)
    plt.imshow(y,vmin=vmin,vmax=vmax,cmap='gray') #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.title('original',fontsize=fontsize)
    # plt.colorbar(shrink= 0.5)

    plt.subplot(142)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_,vmin=vmin,vmax=vmax,cmap='gray')
    # plt.title('denoised',fontsize=fontsize)
    # plt.colorbar(shrink= 0.5)

    plt.subplot(143)
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=vmin,vmax=vmax,cmap='gray')
    # plt.title('removed noise',fontsize=fontsize)


    plt.subplot(144)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(sigma)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.title('predicted sigma',fontsize=fontsize)
    cb=plt.colorbar(shrink=0.5)
    cb.ax.tick_params(labelsize=2)

    plt.tight_layout()
    plt.show()

def show_yxn_gain(y,x_,agc,method):
    fontsize = 16
    noise = y - x_
    if agc:
        y = gain(y, 0.004, 'agc', 0.05, 1)
        x_ = gain(x_, 0.004, 'agc', 0.05, 1)
        noise = gain(noise, 0.004, 'agc', 0.05, 1)

    clip = abs(y).max()
    vmin, vmax = -clip, clip
    import matplotlib.pyplot as plt
    plt.figure(dpi=500,figsize=(6,10)) #(12,9)
    plt.subplot(131)
    plt.imshow(y,vmin=vmin,vmax=vmax,cmap='gray') #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.title('original')
    # plt.colorbar(shrink= 0.5)
    io.savemat(('./noise/shot_hty1.mat'), {'data': y[:, :, np.newaxis]})

    plt.subplot(132)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_,vmin=vmin,vmax=vmax,cmap='gray')
    # plt.title('denoised')
    # plt.colorbar(shrink= 0.5)
    io.savemat(('./noise/shot_' + method + '_dn.mat'), {'data': x_[:, :, np.newaxis]})

    plt.subplot(133)
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=vmin,vmax=vmax,cmap='gray')
    io.savemat(('./noise/shot_' + method + '_n.mat'), {'data': noise[:, :, np.newaxis]})
    # plt.title('removed noise')
    plt.tight_layout()
    plt.show()

###########################
def show_DnNR_3x1(x,y,x_):
    import matplotlib.pyplot as plt
    # ####################
    clip = abs(y).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    fontsize=16
    fig = plt.figure(dpi=500, figsize=(8, 7))  # 16,3

    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2
    # ####################
    snr_x_ = compare_SNR(x, y)
    print(' snr_before= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, y)
    print('psnr_before=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(ssim))
    ####################################################
    # ####################
    snr_x_ = compare_SNR(x, x_)
    print(' snr_after= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_after=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, x_)
    print('ssim_after=', '{:.4f}'.format(ssim))
    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax = plt.subplot(311)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='x', which='major', labelsize=10, labelbottom=False, bottom=False, top=True, labeltop=True)

    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Denoised data', fontsize=fontsize, y=-0.2)
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('./noise/vdn_dn_uns_25.mat'), {'data': x_[:, :, np.newaxis]})

    ax = plt.subplot(312)
    noise = y - x_
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='x', which='major', labelsize=10, labelbottom=False, bottom=False, top=True, labeltop=True)

    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Removed noise', fontsize=fontsize, y=-0.2)
    # plt.colorbar(shrink=0.5)

    ax = plt.subplot(313)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='x', which='major', labelsize=10, labelbottom=False, bottom=False, top=True, labeltop=True)
    plt.imshow(x_ - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Residual', fontsize=fontsize, y=-0.2)
    # plt.colorbar(shrink=0.5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = '20'
    plt.tight_layout()
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})
def show_DnNR_1x3(x,y,x_,method):

    import matplotlib.pyplot as plt
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 12,
    #          }
    method=method
    fontsize=24
    labelfontsize=20
    clip =1 #abs(y).max()  # 显示范围，负值越大越明显  用x好一些
    vmin, vmax = -clip, clip
    fig=plt.figure(dpi=500,figsize=(26, 3))  # 16,3 (26, 3)
    plt.rcParams["font.family"] = "Times New Roman"
    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2
    # ####################
    snr_x_ = compare_SNR(x, y)
    print(' snr_before= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, y)
    print('psnr_before=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(ssim))
    ####################################################
    # ####################
    snr_x_ = compare_SNR(x, x_)
    print(' snr_after= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_after=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, x_)
    print('ssim_after=', '{:.4f}'.format(ssim))
    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax=plt.subplot(131)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)',fontsize=fontsize)
    plt.xlabel('Trace',fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.1)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params (axis='both', which='major', labelsize=labelfontsize, labelbottom = False, bottom=False, top = True, labeltop=True)

    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic) # x_
    plt.title('Denoised section',fontsize=fontsize,y=-0.2)
    plt.colorbar()
    # io.savemat(('../../noise/salt_'+method+'_dn.mat'), {'data': x_[:, :, np.newaxis]})
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(640)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(160)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 641, 639))
    plt.yticks(np.arange(0, 161, 160))
    plt.xlim(0, 641)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(161, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################



    ax=plt.subplot(132)
    noise = y - x_
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.1)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True, labeltop=True)

    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic) #noise
    plt.title('Removed noise section',fontsize=fontsize,y=-0.2)
    plt.colorbar()
    # io.savemat(('../../noise/salt_'+method+'_n.mat'), {'data': noise[:, :, np.newaxis]})
    #../../
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(640)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(160)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 641, 639))
    plt.yticks(np.arange(0, 161, 160))
    plt.xlim(0, 641)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(161, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################



    ax=plt.subplot(133)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.1)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True, labeltop=True)
    plt.imshow(x_ - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Difference section',fontsize=fontsize,y=-0.2)
    plt.colorbar()
    plt.rcParams['font.size'] = '24'
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(640)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(160)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 641, 639))
    plt.yticks(np.arange(0, 161, 160))
    plt.xlim(0, 641)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(161, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################
    plt.rcParams['font.size'] = '24'
    plt.tight_layout()
    # plt.savefig('E:\VIRI\paper\\1stPaperSE\\revise\output\method_compare_output\\'+method+'.png', format='png', dpi=500,
    #             bbox_inches='tight')
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})
def show_DnNR_1x2(x,y,x_):

    import matplotlib.pyplot as plt
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 12,
    #          }
    fontsize=24
    labelfontsize=20
    clip = abs(y).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    fig=plt.figure(dpi=500,figsize=(26, 3))  # 16,3 #figsize=(26, 3)
    plt.rcParams["font.family"] = "Times New Roman"
    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2
    # ####################
    snr_x_ = compare_SNR(x, y)
    print(' snr_before= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, y)
    print('psnr_before=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(ssim))
    ####################################################
    # ####################
    snr_x_ = compare_SNR(x, x_)
    print(' snr_after= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_after=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, x_)
    print('ssim_after=', '{:.4f}'.format(ssim))
    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax=plt.subplot(132)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)',fontsize=fontsize)
    plt.xlabel('Trace',fontsize=fontsize,y=0.2)

    ax.xaxis.set_label_position('top')
    # adjust position of x-axis label
    ax.xaxis.set_label_coords(.5, 1.1)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params (axis='both', which='major', labelsize=labelfontsize, labelbottom = False, bottom=False, top = True, labeltop=True)
    plt.imshow(x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Clean section',fontsize=fontsize,y=-0.2)
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('./noise/salt_clean.mat'), {'data': x[:, :, np.newaxis]})

    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(640)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(160)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 641, 639))
    plt.yticks(np.arange(0, 161, 160))
    plt.xlim(0, 641)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(161, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################

    ax=plt.subplot(131)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    # adjust position of x-axis label
    ax.xaxis.set_label_coords(.5, 1.1)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True, labeltop=True)
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Noisy section',fontsize=fontsize,y=-0.2)
    # plt.colorbar(shrink=0.5)
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(640)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(160)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 641, 639))
    plt.yticks(np.arange(0, 161, 160))
    plt.xlim(0, 641)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(161, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################


    # io.savemat(('./noise/salt_noisy.mat'), {'data': y[:, :, np.newaxis]})

    ax=plt.subplot(133)
    noise = y - x
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    # adjust position of x-axis label
    ax.xaxis.set_label_coords(.5, 1.1)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True, labeltop=True)

    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Noise section',fontsize=fontsize,y=-0.2)
    # io.savemat(('./noise/salt_gt_noise.mat'), {'data': noise[:, :, np.newaxis]})
    # plt.colorbar(shrink=0.5)
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(640)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(160)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 641, 639))
    plt.yticks(np.arange(0, 161, 160))
    plt.xlim(0, 641)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(161, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################
    plt.rcParams['font.size'] = '24'
    plt.tight_layout()
    plt.savefig('E:\VIRI\paper\\1stPaperSE\\revise\output\method_compare_output\salt.png', format='png',
                dpi=500,
                bbox_inches='tight')
    plt.show()


def show_DnNR_f_1x3(x,y,x_,method):

    import matplotlib.pyplot as plt
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 12,
    #          }
    method=method
    fontsize=24
    labelfontsize=20
    clip =1 #abs(x).max()  # 显示范围，负值越大越明显  用x好一些
    vmin, vmax = -clip, clip
    fig=plt.figure(dpi=100,figsize=(18, 5))  # 16,3 (26, 3)dpi=500 (18,5)


    plt.rcParams["font.family"] = "Times New Roman"
    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2
    # ####################
    snr_x_ = compare_SNR(x, y)
    print(' snr_before= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, y)
    print('psnr_before=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(ssim))
    ####################################################
    # ####################
    snr_x_ = compare_SNR(x, x_)
    print(' snr_after= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_after=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, x_)
    print('ssim_after=', '{:.4f}'.format(ssim))
    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax=plt.subplot(131)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)',fontsize=fontsize)
    plt.xlabel('Trace',fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.05)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params (axis='both', which='major', labelsize=labelfontsize, labelbottom = False, bottom=False, top = True, labeltop=True)
    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Denoised section',fontsize=fontsize,y=-0.1)

    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/salt_'+method+'_dn.mat'), {'data': x_[:, :, np.newaxis]})
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(480)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(384)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax1 = plt.gca()
    # ax为两条坐标轴的实例
    ax1.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax1.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 481, 479))
    plt.yticks(np.arange(0, 385, 384), ['492', '876'])
    plt.xlim(0, 481)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(385, 0) #(385+492, 0+492) (385, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################

    ax=plt.subplot(132)
    noise = y - x_
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.05)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True, labeltop=True)

    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Removed noise section',fontsize=fontsize,y=-0.1)
    # plt.colorbar(shrink=0.5)
    # io.savemat(('../../noise/salt_'+method+'_n.mat'), {'data': noise[:, :, np.newaxis]})
    #../../
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(480)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(384)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 481, 479))
    plt.yticks(np.arange(0, 385, 384), ['492', '876'])
    plt.xlim(0, 481)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(385, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################



    ax=plt.subplot(133)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.05)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True, labeltop=True)
    plt.imshow(x_ - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Difference section',fontsize=fontsize,y=-0.1)
    # plt.colorbar(shrink=0.8)

    plt.rcParams['font.size'] = '24'
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(480)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(384)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 481, 479))
    plt.yticks(np.arange(0, 385, 384),['492','876'])
    plt.xlim(0, 481)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(385, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################
    plt.rcParams['font.size'] = '24'
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0,hspace=0)

    # plt.savefig('E:\VIRI\paper\\1stPaperSE\\revise\output\method_compare_output\\'+method+'.png', format='png', dpi=500,
    #             bbox_inches='tight')
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})

def show_DnNR_f_1x3_(y,x_,method):

    import matplotlib.pyplot as plt
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 12,
    #          }
    method=method
    fontsize=24
    labelfontsize=20
    clip =abs(y).max() #1 #abs(x).max()  # 显示范围，负值越大越明显  用x好一些
    clip = 0.2
    vmin, vmax = -clip, clip
    fig=plt.figure(dpi=300,figsize=(10, 6))  # 16,3 (26, 3)dpi=500 (18,5)


    plt.rcParams["font.family"] = "Times New Roman"
    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2
    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax=plt.subplot(131)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap='gray') #'gray' plt.cm.seismic
    # plt.title('Noisy section', fontsize=fontsize, y=-0.1)

    ax=plt.subplot(132)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax,cmap='gray')#,
    # plt.title('Denoised section',fontsize=fontsize,y=-0.1)

    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/salt_'+method+'_dn.mat'), {'data': x_[:, :, np.newaxis]})
    #############


    ax=plt.subplot(133)
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



def show_DnNSimi_f_1x3(x, y, x_,simi, method):
    import matplotlib.pyplot as plt
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 12,
    #          }
    method = method
    fontsize = 24
    labelfontsize = 20
    clip = 1  # abs(x).max()  # 显示范围，负值越大越明显  用x好一些
    vmin, vmax = -clip, clip
    fig = plt.figure(dpi=100, figsize=(18, 5))  # 16,3 (26, 3)dpi=500 (18,5)
    plt.rcParams["font.family"] = "Times New Roman"
    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2
    # ####################
    snr_x_ = compare_SNR(x, y)
    print(' snr_before= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, y)
    print('psnr_before=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(ssim))
    ####################################################
    # ####################
    snr_x_ = compare_SNR(x, x_)
    print(' snr_after= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_after=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, x_)
    print('ssim_after=', '{:.4f}'.format(ssim))
    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax = plt.subplot(131)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.05)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True,
                    labeltop=True)

    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Denoised section', fontsize=fontsize, y=-0.1)
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/salt_'+method+'_dn.mat'), {'data': x_[:, :, np.newaxis]})
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(480)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(384)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax1 = plt.gca()
    # ax为两条坐标轴的实例
    ax1.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax1.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 481, 479))
    plt.yticks(np.arange(0, 385, 384), ['492', '876'])
    plt.xlim(0, 481)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(385, 0)  # (385+492, 0+492) (385, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################

    ax = plt.subplot(132)
    noise = y - x_
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(.5, 1.05)
    ax.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True,
                    labeltop=True)

    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('Removed noise section', fontsize=fontsize, y=-0.1)
    # plt.colorbar(shrink=0.5)
    # io.savemat(('../../noise/salt_'+method+'_n.mat'), {'data': noise[:, :, np.newaxis]})
    # ../../
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(480)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(384)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 481, 479))
    plt.yticks(np.arange(0, 385, 384), ['492', '876'])
    plt.xlim(0, 481)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(385, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################

    ax3 = plt.subplot(133)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.ylabel('Time(ms)', fontsize=fontsize)
    plt.xlabel('Trace', fontsize=fontsize)
    ax3.xaxis.set_label_position('top')
    ax3.xaxis.set_label_coords(.5, 1.05)
    ax3.yaxis.set_label_coords(-.02, .5)
    plt.tick_params(axis='both', which='major', labelsize=labelfontsize, labelbottom=False, bottom=False, top=True,
                    labeltop=True)
    plt.imshow(simi, vmin=0,vmax=1,cmap=plt.cm.jet)
    plt.title('Local similarity', fontsize=fontsize, y=-0.1)
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # ax = plt.gca()
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # plt.colorbar(shrink=0.8)

    # #前三个子图总宽度为原来的0.9
    # fig.subplots_adjust(right=0.9)
    # #colorbar 左下宽高
    # l=0.92
    # b=0.12
    # w=0.015
    # h=1-2*b
    # #对应l,b,w,h.设置colorbar位置
    # rect=[l,b,w,h]
    # cbar_ax=fig.add_axes(rect)
    # plt.colorbar(cax=cbar_ax)

    # plt.colorbar(shrink=0.5)
    plt.rcParams['font.size'] = '24'
    #############
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(480)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(384)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xticks(np.arange(1, 481, 479))
    plt.yticks(np.arange(0, 385, 384), ['492', '876'])
    plt.xlim(0, 481)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(385, 0)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    #######################
    plt.rcParams['font.size'] = '24'
    # plt.tight_layout()
    # plt.savefig('E:\VIRI\paper\\1stPaperSE\\revise\output\method_compare_output\\' + method + '.png', format='png',
    #             dpi=500,
    #             bbox_inches='tight')
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})

def show_DnNSimi_f_1x3_(y, x_,simi, method):

    import matplotlib.pyplot as plt
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 12,
    #          }
    method=method
    fontsize=24
    labelfontsize=20
    clip =1 #abs(x).max()  # 显示范围，负值越大越明显  用x好一些
    vmin, vmax = -clip, clip
    fig=plt.figure(dpi=300,figsize=(10, 6))  #


    plt.rcParams["font.family"] = "Times New Roman"
    # fig.suptitle("FXDECONV", y=1.05, fontsize=18) #总标题y=-0.2

    ####################################################
    # f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax=plt.subplot(131)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap='gray') #'gray'  plt.cm.seismic
    plt.title('Denoised section',fontsize=fontsize,y=-0.1)

    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/salt_'+method+'_dn.mat'), {'data': x_[:, :, np.newaxis]})
    #############


    ax=plt.subplot(132)
    noise = y - x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap='gray')
    plt.title('Removed noise section', fontsize=fontsize, y=-0.1)
    # plt.colorbar(shrink=0.5)
    # io.savemat(('../../noise/salt_'+method+'_n.mat'), {'data': noise[:, :, np.newaxis]})
    #../../

    ax=plt.subplot(133)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(simi, vmin=0, vmax=1, cmap=plt.cm.jet)
    plt.title('Local similarity', fontsize=fontsize, y=-0.1)
    # plt.colorbar(shrink=0.8)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0,hspace=0)

    # plt.savefig('E:\VIRI\paper\\1stPaperSE\\revise\output\method_compare_output\\'+method+'.png', format='png', dpi=500,
    #             bbox_inches='tight')
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})
