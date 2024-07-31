from crip.io import imreadRaws, imreadTiffs, imreadDicoms, imwriteRaw, imwriteTiff, imreadTiff
from crip.postprocess import huNoRescale
import os
import numpy as np
from tqdm import trange
import cv2

def genVoxFileBM(output_path,mat1,mat2,VoxelXYSize,VoxelZSize):
    assert len(mat1) == len(mat2), "mat1 should have same length as mat2."
    VoxX = mat1.shape[1]
    VoxY = mat1.shape[2]
    VoxZ = mat1.shape[0]

    mat1 = mat1.flatten()
    mat2 = mat2.flatten()

    SavePath = output_path
    f = open(SavePath, 'w')
    f.write('[SECTION VOXELS phantomER]\n')
    f.write(str(VoxX) + ' ' + str(VoxY) + ' ' + str(VoxZ) + ' No. OF VOXELS IN X,Y,Z\n')
    f.write(str(VoxelXYSize) + ' ' + str(VoxelXYSize) + ' ' + str(VoxelZSize) + ' VOXEL SIZE (cm) ALONG X,Y,Z\n')
    f.write('1 COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n')
    f.write('2 COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n')
    f.write('0 BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n')
    f.write('[END OF VXH SECTION]\n')

    for i in trange(mat1.shape[0], ncols=80):
        if mat1[i] <= 0.1 and mat2[i] <= 0.1:
            f.write('{} {}\n'.format('1', '0.00120479'))
        elif mat1[i] < mat2[i]:
            f.write('{} {}\n'.format('2', str(mat2[i])))
        elif mat1[i] >= mat2[i]:
            f.write('{} {}\n'.format('3', str(mat1[i] * 1.92)))

    print('{} is done.'.format(output_path))

def genVoxFileCT(output_path,vol,VoxelXYSize,VoxelZSize):
    VoxX = vol.shape[1]
    VoxY = vol.shape[2]
    VoxZ = vol.shape[0]

    vol = vol.flatten()

    SavePath = output_path
    f = open(SavePath, 'w')
    f.write('[SECTION VOXELS phantomER]\n')
    f.write(str(VoxX) + ' ' + str(VoxY) + ' ' + str(VoxZ) + ' No. OF VOXELS IN X,Y,Z\n')
    f.write(str(VoxelXYSize) + ' ' + str(VoxelXYSize) + ' ' + str(VoxelZSize) + ' VOXEL SIZE (cm) ALONG X,Y,Z\n')
    f.write('1 COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n')
    f.write('2 COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n')
    f.write('0 BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n')
    f.write('[END OF VXH SECTION]\n')

    for i in trange(vol.shape[0], ncols=80):
        if vol[i] == 0:
            f.write("{} {}\n".format("1", "0.00120479"))
        else:
            f.write("{} {}\n".format("2", "1"))
        # f.write("{} {}\n".format("1", "0.00120479"))
    print('{} is done.'.format(output_path))

def rawVol2Vox(input_path,output_path,style,h,w,NSlice,VoxelXYSize,VoxelZSize):
    if style == "basis_material":
        for dir in os.listdir(input_path):
            mat1_path = os.path.join(input_path, dir, "bone")
            mat2_path = os.path.join(input_path, dir, "water")
            mat1 = imreadRaws(mat1_path, h, w, dtype=np.float32, nSlice=NSlice)
            mat2 = imreadRaws(mat2_path, h, w, dtype=np.float32, nSlice=NSlice)

            output_path_ = os.path.join(output_path, f"{dir}.vox")
            genVoxFileBM(output_path_,mat1,mat2,VoxelXYSize,VoxelZSize)
    elif style == "ctimg":
        for dir in os.listdir(input_path):
            img_path = os.path.join(input_path, dir)
            imgs = imreadRaws(img_path, h, w, dtype=np.float32, nSlice=NSlice)

            if imgs.any()< 0:
                imgs = huNoRescale(imgs)
            output_path_ = os.path.join(output_path, f"{dir}.vox")
            genVoxFileCT(output_path_, imgs, VoxelXYSize, VoxelZSize)

def tiffVol2Vox(input_path,output_path,style,VoxelXYSize,VoxelZSize):
    if style == "basis_material":
        for dir in os.listdir(input_path):
            mat1_path = os.path.join(input_path, dir, "bone")
            mat2_path = os.path.join(input_path, dir, "water")
            mat1 = imreadTiffs(mat1_path, dtype=np.float32)
            mat2 = imreadTiffs(mat2_path, dtype=np.float32)

            output_path_ = os.path.join(output_path, f"{dir}.vox")
            genVoxFileBM(output_path_, mat1, mat2, VoxelXYSize, VoxelZSize)

    elif style == "ctimg":
        for dir in os.listdir(input_path):
            img_path = os.path.join(input_path, dir)
            imgs = imreadTiffs(img_path,dtype=np.float32)

            if imgs.any()< 0:
                imgs = huNoRescale(imgs)
            output_path_ = os.path.join(output_path, f"{dir}.vox")
            genVoxFileCT(output_path_, imgs, VoxelXYSize, VoxelZSize)

def dicomVol2Vox(input_path,output_path,style,VoxelXYSize,VoxelZSize):
    if style == "ctimg":
        for dir in os.listdir(input_path):
            img_path = os.path.join(input_path,dir)
            imgs = imreadDicoms(img_path,dtype=np.float32,attrs={'SamplesPerPixel': 1, 'PhotometricInterpretation': 1})[::2, :, :]
            print(imgs.shape)

            if imgs.any()< 0:
                imgs = huNoRescale(imgs)

            output_path_ = os.path.join(output_path, f"{dir}_2.vox")
            genVoxFileCT(output_path_,imgs,VoxelXYSize,VoxelZSize)

def Vox2Vol(input_path,output_path,img_fmt):
    vox = open(input_path, "r")
    vox.readline()
    X, Y, Z = map(int, vox.readline().split(" ")[:3])
    sizeX, sizeY, sizeZ = map(float, vox.readline().split(" ")[:3])
    for i in range(4):
        vox.readline()

    vol = np.zeros([X, Y, Z]).flatten()
    water = np.zeros([X, Y, Z]).flatten()
    bone = np.zeros([X, Y, Z]).flatten()
    num = 0
    while True:
        con = vox.readline().strip().split(" ")
        if con[0] == "":
            break
        elif con[0] == "2":
            vol[num] = float(con[1])
            water[num] = float(con[1])

        # elif con[0] == "3":
        #     vol[num] = float(con[1]) / 1.92
        #     bone[num] = float(con[1]) / 1.92
        num += 1

    mat_vol = vol.reshape(Z, Y, X)
    mat_water = water.reshape(Z, Y, X)
    mat_bone = bone.reshape(Z, Y, X)

    if img_fmt == "raw":
        imwriteRaw(mat_water,os.path.join(output_path,"mat_water.raw"),dtype=np.float32)
        imwriteRaw(mat_bone, os.path.join(output_path,"mat_bone.raw"), dtype=np.float32)
        imwriteRaw(mat_vol, os.path.join(output_path,"mat_vol.raw"), dtype=np.float32)
    elif img_fmt == "tiff":
        imwriteTiff(mat_water,os.path.join(output_path,"mat_water.tif"),dtype=np.float32)
        imwriteTiff(mat_bone,os.path.join(output_path,"mat_bone.tif"),dtype=np.float32)
        imwriteTiff(mat_vol, os.path.join(output_path,"mat_vol.tif"), dtype=np.float32)


if __name__ == "__main__":
    input_path = r"./Data/waterNew.tif"
    output_path = "./runs/waterNew.vox"
    h, w = 512, 512
    NSlice = 1
    VoxelXYSize = 0.042
    VoxelZSize = 0.042
    imgs = imreadTiff(input_path)
    genVoxFileCT(output_path,imgs,VoxelXYSize, VoxelZSize)
    # rawVol2Vox(input_path,output_path,style="ctimg",h=512,w=512,NSlice=1,VoxelXYSize=VoxelXYSize,VoxelZSize=VoxelZSize)
    # tiffVol2Vox(input_path,output_path,style="ctimg",VoxelXYSize=VoxelXYSize,VoxelZSize=VoxelZSize)
    # dicomVol2Vox(input_path,output_path,style="ctimg",VoxelXYSize=VoxelXYSize,VoxelZSize=VoxelZSize)

    # Vox2Vol(input_path,output_path,img_fmt="tiff")