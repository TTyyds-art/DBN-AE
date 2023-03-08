import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import as_tensor
import numpy as np


def data_loader(the_path=None, in_num=11, out_num=14):
    """
    in_num: the input number of MLP, ie, the num of load buses
    out_num: the output number of MLP, ie, the num of buses
    """
    in_num = int(in_num)  # the input length, which is from om, so we need to provide.
    if the_path is None:
        file_paths = ['/home/ubuntu-h/PycharmProjects/Auto_model_Ps2Dl/Power_Relia_Proj/data',
                      '/home/huzuntao/PycharmProjects/Auto_model_Ps2Dl/Power_Relia_Proj/data']
        for path_ in file_paths:
            if os.path.exists(path_):
                the_path = path_
        the_path = '/home/huzuntao/PycharmProjects/Auto_model_Ps2Dl/Power_Relia_Proj/data/bus14_2nd_2022_8_21/22_34_20.npy'
    train_data = np.load(the_path)
    num_samples = train_data.shape[0]
    real_inps = train_data[:num_samples, :in_num].real
    imag_inps = train_data[:num_samples, :in_num].imag

    oups = train_data[:num_samples, in_num:]     # complex number
    v_oups, s_oups = oups[:, :out_num], oups[:, out_num:]
    v_ans, v_ms = np.angle(v_oups), np.abs(v_oups)
    p_oups, q_oups = s_oups.real, s_oups.imag

    inps = as_tensor(np.hstack((real_inps, imag_inps)), dtype=torch.float32)
    oups = as_tensor(np.hstack((v_ans, v_ms, p_oups, q_oups)), dtype=torch.float32)
    # dataset = TensorDataset(inps, oups)
    return inps, oups


if __name__ == "__main__":
    _path_30 = '/home/huzuntao/PycharmProjects/Auto_model_Ps2Dl/Power_Relia_Proj/data/bus30_1st_2022_8_22/21_57_26.npy'
    dataset = data_loader(the_path=_path_30, in_num=20, out_num=30)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
