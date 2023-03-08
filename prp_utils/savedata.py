import numpy as np
import os
import torch
from pathlib import Path
from torch import nn
import time


def save_numpy_data(data, name=None, temp=True):

    """每天作为一个文件夹，name+年_月_日为其名； 每个文件是时_分_秒命名
    return: the file-path
    """

    ticks = time.localtime()
    assert isinstance(data, np.ndarray), "the data is not a numpy array"

    if name is None:
        name = str(ticks.tm_year) + '_' + str(ticks.tm_mon) + '_' + str(ticks.tm_mday)
    elif isinstance(name, str):
        name = name + '_' + str(ticks.tm_year) + '_' + str(ticks.tm_mon) + '_' + str(ticks.tm_mday)
    else:
        name = str(name) + '_' + str(ticks.tm_year) + '_' + str(ticks.tm_mon) + '_' + str(ticks.tm_mday)

    if temp is True:
        name = 'Temp_' + name

    path_current = os.getcwd()
    path_temp = path_current.rsplit("/", 1)
    while path_temp[1] != 'DBN-AE':
        path_temp = path_temp[0].rsplit("/", 1)
    path_PRP = path_temp[0] + "/" + path_temp[1]
    dir_path = path_PRP + '/data/' + name
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    file_name = str(ticks.tm_hour) + '_' + str(ticks.tm_min) + '_' + str(ticks.tm_sec)

    np.save(dir_path + '/' + file_name, data)
    # give a flag to make dir
    print(f"The data will be save in the dir: {dir_path + '/' + file_name}")
    time.sleep(1)
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            print(os.path.join(root, name))

    return dir_path + '/' + file_name + '.npy'

    # os.remove(dir_path)
    # print(f"removed!")


def save_pytorch_params(model_name, name_dir=None, temp=False):
    ticks = time.localtime()
    assert isinstance(model_name, nn.Module), "the data is not a torch object"

    if name_dir is None:
        name_dir = str(ticks.tm_year) + '_' + str(ticks.tm_mon) + '_' + str(ticks.tm_mday)
    elif isinstance(name_dir, str):
        name_dir = name_dir + '_' + str(ticks.tm_year) + '_' + str(ticks.tm_mon) + '_' + str(ticks.tm_mday)
    else:
        name_dir = str(name_dir) + '_' + str(ticks.tm_year) + '_' + str(ticks.tm_mon) + '_' + str(ticks.tm_mday)

    if temp is True:
        name_dir = 'Temp_' + name_dir

    path_current = os.getcwd()
    path_temp = path_current.rsplit("/", 1)
    while path_temp[1] != 'Power_Relia_Proj':
        path_temp = path_temp[0].rsplit("/", 1)
    path_PRP = path_temp[0] + "/" + path_temp[1]

    dir_path = path_PRP + '/model_params/' + name_dir
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    file_name = str(ticks.tm_hour) + '_' + str(ticks.tm_min) + '_' + str(ticks.tm_sec)

    torch.save(model_name.state_dict(), dir_path + '/' + file_name)

    print(f"The params will be save in the dir: {dir_path + '/' + file_name}")
    time.sleep(1)
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name_d in files:
            print(os.path.join(root, name_d))


def del_temp_dir_with_data(dirname_start='Temp_'):
    file_path = ['/home/ubuntu-h/PycharmProjects/Auto_model_Ps2Dl/Power_Relia_Proj/data',
                 '/home/huzuntao/PycharmProjects/Auto_model_Ps2Dl/Power_Relia_Proj/data']
    for path_ in file_path:
        if os.path.exists(path_):
            print(f"path_: {path_}")
            for root, dirs, _ in os.walk(path_, topdown=False):
                print(f"main root: {root}")
                for name in dirs:
                    if str.startswith(name, dirname_start):
                        for root_, _, files in os.walk(os.path.join(root, name), topdown=False):
                            print(f'sub root_: {root_}; files: {files}')
                            for pathf in [os.path.join(root_, file) for file in files]:
                                os.remove(pathf)
                        print(f"joined path: {os.path.join(root, name)}; root: {root}")
                        os.removedirs(os.path.join(root, name))


if __name__ == '__main__':
    da = np.zeros(shape=[3])
    path_save = save_numpy_data(da, name='test', temp=False)
    print(f"path: {path_save}")
    # del_temp_dir_with_data()
