import os
import scipy.io as sio
import spectral
from spectral import *

dataset_path = os.path.join('W:\data')
data = sio.loadmat(os.path.join(dataset_path, 'data.mat'))['data']   # Botswana   Paviau
spectral.settings.WX_GL_DEPTH_SIZE = 5
view_cube(data, bands=[35, 25, 18])