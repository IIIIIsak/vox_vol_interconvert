import scipy.io as scio
import numpy as np
from tqdm import trange
import pandas as pd


def gen_vox_file(sex, scale, savePath, matPath):
    # read map_MCGPU
    map_MCGPU = pd.read_excel('./map.xlsx')
    organ_id = map_MCGPU['organ_ID'].values
    material_id = map_MCGPU['MCGPU_material_ID'].values
    density = map_MCGPU[f'{sex}_density'].values
    material_map = dict(zip(organ_id, material_id))
    density_map = dict(zip(organ_id, density))

    # read Phantom data
    mat_file = matPath
    mat = scio.loadmat(mat_file)
    name = list(filter(lambda x: x[:2] != '__', mat.keys()))[0]
    mat = mat[name]
    mat = np.flip(mat, axis=2)
    print(mat.shape)

    nx = mat.shape[0]
    ny = mat.shape[1]
    nz = mat.shape[2]
    dx = 0.05 * scale
    dy = 0.05 * scale
    dz = 0.05 * scale
    phantom = np.array(mat, dtype='float')
    phantom = phantom.flatten(order='F')
    total_voxels = phantom.size

    # assign material
    f = open(savePath, 'w')
    f.write('[SECTION VOXELS phantomER]\n')
    f.write(str(nx) + ' ' + str(ny) + ' ' + str(nz) + ' No. OF VOXELS IN X,Y,Z\n')
    f.write(str(dx) + ' ' + str(dy) + ' ' + str(dz) + ' VOXEL SIZE (cm) ALONG X,Y,Z\n')
    f.write('1 COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n')
    f.write('2 COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n')
    f.write('0 BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n')
    f.write('[END OF VXH SECTION]\n')

    for i in trange(total_voxels, ncols=80):
        f.write(str(material_map[phantom[i]]) + ' ' + str(density_map[phantom[i]]) + '\n')

    f.close()

def gen_air_vox_file(sex, scale, savePath, matPath):

    # read Phantom data
    mat_file = matPath
    mat = scio.loadmat(mat_file)
    name = list(filter(lambda x: x[:2] != '__', mat.keys()))[0]
    mat = mat[name]
    mat = np.flip(mat, axis=2)
    print(mat.shape)

    nx = mat.shape[0]
    ny = mat.shape[1]
    nz = mat.shape[2]
    dx = 0.0425 * scale
    dy = 0.0425 * scale
    dz = 0.0425 * scale
    phantom = np.array(mat, dtype='float')
    phantom = phantom.flatten(order='F')
    total_voxels = phantom.size

    # assign material
    f = open(savePath, 'w')
    f.write('[SECTION VOXELS phantomER]\n')
    f.write(str(nx) + ' ' + str(ny) + ' ' + str(nz) + ' No. OF VOXELS IN X,Y,Z\n')
    f.write(str(dx) + ' ' + str(dy) + ' ' + str(dz) + ' VOXEL SIZE (cm) ALONG X,Y,Z\n')
    f.write('1 COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n')
    f.write('2 COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n')
    f.write('0 BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n')
    f.write('[END OF VXH SECTION]\n')

    for i in trange(total_voxels, ncols=80):
        f.write('1' + ' ' + "0.00120479" + '\n')

    f.close()

for kind in ['chest_no_arm']:  # 'abdomen_no_arm', 'chest_no_arm', 'head'
    for sex in ['Female']:
        for scale in [1]:  # 0.8, 1, 1.2
            print(kind, sex, scale)
            MatFilePath = f'./{sex}/data_{kind}.mat'
            SavePath = f'./Vox/air_{sex.lower()}_{kind}_{scale}.vox'
            gen_air_vox_file(sex, scale, SavePath, MatFilePath)
