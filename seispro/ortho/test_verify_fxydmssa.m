clc;clear;close all;
%% 3D
load('datas.mat');

d11=fxydmssa(dn,0,120,0.004,3,3,0);
norm(d1(:)-d11(:))

figure; yc_imagesc([dn(:,:),d1(:,:),dn(:,:)-d1(:,:)]);
figure; yc_imagesc([d1(:,:),d11(:,:),d1(:,:)-d11(:,:)],90);

yc_snr(d0,dn,2)
yc_snr(d0,d1,2)
yc_snr(d0,d11,2)

clc;clear;close all;
%% 2D
load('datas2d.mat');
d11=fxydmssa(dn,0,120,0.004,3,3,0);
norm(d1(:)-d11(:))

figure; yc_imagesc([dn(:,:),d1(:,:),dn(:,:)-d1(:,:)]);
figure; yc_imagesc([d1(:,:),d11(:,:),d1(:,:)-d11(:,:)],90);

yc_snr(d0,dn,2)
yc_snr(d0,d1,2)
yc_snr(d0,d11,2)

