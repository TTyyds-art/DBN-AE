import math
import copy
import numpy as np
from powernets.gridinit import grid_init
import pandapower.networks as nw  # for case name
from prp_utils.savedata import save_numpy_data, del_temp_dir_with_data


def produce_train_data(num_data=1, case_=None, devia_delta=0.2):
    """
    :param
    num_data: the number of samples
    case_: the case generated from PandaPower
    devia_delta : set the Pload devilation
    produce OPF samples with particular numbers
    """
    i = 1
    samples = []
    if case_ is not None:
        case_new = case_
    else:
        _, case_new = grid_init()
    if case_new.bus.shape[0] == 14 or case_new.bus.shape[0] == 30:
        case_new.gen['vm_pu'] = 1.00  # 有些模型不能跑通，需要做一些调整
    l_p = copy.deepcopy(case_new.load['p_mw'])

    while i <= num_data:
        devia_rate = [np.random.uniform(1 - devia_delta, 1 + devia_delta) for i in
                      range(len(case_new.load['p_mw']))]  # 确定某一个量,偏移一点
        p_load_ = l_p * devia_rate
        try:
            print(f"p load: {p_load_[1]}; l_p:{l_p[1]}: devia_rate:{devia_rate[1]}")
            om, l_idx = grid_init(case_name=case_new, p_load=p_load_)
        except:
            print('Not converged!')
        else:
            i += 1
            sample = _train_OPFdata(net=case_new)
            samples.append(sample)
    if isinstance(samples, list) and len(samples) > 0:
        return np.array(samples)
    else:
        raise "There is no feasible solution of this OPF !"


def produce_test_data(in_sequ=None, case_=None, devia_delta=0.2):
    """
    :param
    num_data: the number of samples
    case_: the case generated from PandaPower
    devia_delta : set the Pload devilation
    produce OPF samples with particular numbers
    """
    i = 0
    samples = []
    if case_ is not None:
        case_new = case_
    else:
        _, case_new = grid_init()
    if case_new.bus.shape[0] == 14 or case_new.bus.shape[0] == 30:
        case_new.gen['vm_pu'] = 1.00  # 有些模型不能跑通，需要做一些调整
    l_p = copy.deepcopy(case_new.load['p_mw'])

    ind_max = np.argmax(l_p)
    if ind_max == 0:
        ind_max += 1
    elif ind_max == len(l_p)-1:
        ind_max -= 1
    a = np.linspace(1 - devia_delta, 1 + devia_delta, 100).reshape([100, 1])
    ll = np.ones([100, 1]) * (1 - devia_delta)
    uu = np.ones([100, 1]) * (1 + devia_delta)
    if in_sequ is None:
        in_sequ = ['uu', 'a', 'll']
    dict_devia = {'a': a, 'll': ll, 'uu': uu}

    # tot_devia = np.hstack([dict_devia[in_sequ[0]] * np.ones([1, ind_max])
    #                           , dict_devia[in_sequ[1]]
    #                           , dict_devia[in_sequ[2]] * np.ones([1, len(l_p) - ind_max - 1])])
    tot_devia = np.ones([1, len(l_p)]) * a

    while i < len(tot_devia):
        devia_rate = tot_devia[i]
        p_load_ = l_p * devia_rate

        try:
            print(f"p load: {p_load_[0]}; l_p:{l_p[1]}: devia_rate:{devia_rate[2]}")
            om, l_idx = grid_init(case_name=case_new, p_load=p_load_)
        except:
            print('Not converged!')
        else:
            i += 1
            sample = _train_OPFdata(net=case_new)
            samples.append(sample)
    if isinstance(samples, list) and len(samples) > 0:
        return np.array(samples)
    else:
        raise "There is no feasible solution of this OPF !"


def produce_added_data(load_data=None, case_name=None):
    """
    用来计算新添加的负载数据 load data
        :param
        load_data: the load data from finding by the front net
        case_: the case generated from PandaPower
        devia_delta : set the Pload devilation
        :return
        produce OPF samples for added load data
        """
    global case_new
    i = 0
    samples = []
    if case_name is None:
        if len(load_data[0]) == 22:
            case_new = nw.case14()
            case_new.gen['vm_pu'] = 1.00  # 有些模型不能跑通，需要做一些调整
        elif 22 < len(load_data[0]) < 60:
            case_new = nw.case_ieee30()
            case_new.gen['vm_pu'] = 1.00
        else:
            case_new = None
    else:
        case_new = case_name

    if case_new.bus.shape[0] == 14 or case_new.bus.shape[0] == 30:
        case_new.gen['vm_pu'] = 1.00  # 有些模型不能跑通，需要做一些调整

    num_load = case_new.load.shape[0]
    if len(load_data.shape) == 1:  # deal with only one input situation
        load_data = load_data.to_numpy().reshape([1, -1])

    while i < len(load_data):
        try:
            p_load_ = load_data[i]
            print(f"index: {i}")
            om, l_idx = grid_init(case_name=case_new, p_load=p_load_[:num_load])
        except:
            i += 1
            print('Not converged!')
        else:
            i += 1
            sample = _train_OPFdata(net=case_new)
            samples.append(sample)
    if isinstance(samples, list) and len(samples) > 0:
        return np.array(samples)
    else:
        raise "There is no feasible solution of this OPF !"


def _train_OPFdata(net):
    """save the Sload, V, Sgen in sample array after calculating the OPF of  the net."""
    print('status:', net["OPF_converged"])

    s_load = net.res_load['p_mw'] + 1j * net.res_load['q_mvar']

    vm_pu = net.res_bus["vm_pu"]
    va_ra = net.res_bus["va_degree"] * math.pi / 180  # 转化成弧度
    v_complx = vm_pu * np.exp(1j * va_ra)

    p_extra = net.res_ext_grid['p_mw']
    q_extra = net.res_ext_grid['q_mvar']
    s_extra = p_extra + 1j * q_extra
    p_gen = net.res_gen['p_mw']
    q_gen = net.res_gen['q_mvar']
    s_gen = p_gen + 1j * q_gen

    l_l, l_e, l_g, l_v = len(s_load), len(s_extra), len(s_gen), len(v_complx)
    num_col = l_l + l_e + l_g + l_v
    sample = np.empty(shape=[num_col], dtype=complex)
    sample[:l_l] = s_load
    sample[l_l:l_v + l_l] = v_complx
    sample[l_v + l_l:l_v + l_l + l_e] = s_extra
    sample[l_v + l_l + l_e:] = s_gen

    return sample


if __name__ == "__main__":
    # case_new = nw.case_ieee30()
    case_new = nw.case14()
    # case_new = nw.case118()
    # case_new.gen['vm_pu'] = 1.00
    # case_new.
    # samples_train = produce_train_data(num_data=500, case_=case_new, devia_delta=0.2)
    # NOTE: turn off the save module
    # save_numpy_data(samples_train, name='bus_14_500_1st', temp=False)

    samples_tr_n_te = produce_test_data( case_=case_new, devia_delta=0.2)

    # NOTE: turn off the save module
    save_numpy_data(samples_tr_n_te, name='14-bus-0.2', temp=False)
    # samples_tr_n_te = produce_test_data(in_sequ=['ll', 'a', 'uu'], case_=case_new, devia_delta=0.2)
    # NOTE: turn off the save module
    # save_numpy_data(samples_tr_n_te, name='llauu', temp=False)
    # samples_tr_n_te = produce_test_data(in_sequ=['ll', 'a', 'll'], case_=case_new, devia_delta=0.2)
    # NOTE: turn off the save module
    # save_numpy_data(samples_tr_n_te, name='llall', temp=False)
    # del_temp_dir_with_data()
    # sample: [loads, voltage, gens]

    # case_new.gen['vm_pu'] = 1.01  # 有些模型不能跑通，需要做一些调整
    # load_X = case_new.load['p_mw']
    # load_X_ = np.load('add.npy')
    # samples_train = produce_added_data(load_data=load_X_[:1])
    # print(samples_train[:, 5])
