import pandapower as pp
import pandapower.networks as nw
def grid_init(case_name=None, p_load=None):
    """

    :param case_name: the net object from PandaPower
    :param p_load: numpy ndarray
    :return: om, net
    """
    if case_name is not None:
        net = case_name
        if p_load is not None:
            net.load['p_mw'] = p_load
        obm_, _ = pp.runopp(net, delta=1e-16)
        load_index = net.load["bus"]

        return obm_, net

    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, vn_kv=110.)
    bus2 = pp.create_bus(net, vn_kv=110.)
    bus3 = pp.create_bus(net, vn_kv=110.)
    bus4 = pp.create_bus(net, vn_kv=110.)
    # bus5 = pp.create_bus(net, vn_kv=110.)

    # create 220/110 kV transformer
    pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')

    # create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type="149-AL1/24-ST1A 110.0")
    pp.create_line(net, bus4, bus2, length_km=40., std_type="149-AL1/24-ST1A 110.0")

    # create loads
    pp.create_load(net, bus2, p_mw=60., controllable=False)
    pp.create_load(net, bus3, p_mw=70., controllable=False)
    pp.create_load(net, bus4, p_mw=25., controllable=False)

    # create generators
    eg = pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000)
    g0 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.00, controllable=True)
    g1 = pp.create_gen(net, bus4, p_mw=100, min_p_mw=0, max_p_mw=100, vm_pu=1.00, controllable=True)

    costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
    costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
    costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

    net.bus["min_vm_pu"] = 0.96
    net.bus["max_vm_pu"] = 1.04
    net.line["max_loading_percent"] = 100
    if p_load is not None:
        net.load['p_mw'] = p_load
    obm_, _ = pp.runopp(net, delta=1e-16)
    load_index = net.load["bus"]

    return obm_, net

def choose_modify_case(name_idx:str = '14'):
    import pandapower as pp
    import pandapower.networks as nw

    if name_idx == '14':
        case_name = nw.case14()
        case_name.gen['vm_pu'] = 1.00
    elif name_idx == '30':
        case_name = nw.case_ieee30()
        case_name.gen['vm_pu'] = 1.00
    elif name_idx == '4':
        _, case_name = grid_init()
    elif name_idx == '118':
        case_name = nw.case118()

    return case_name


if __name__ == '__main__':
    # case_name = nw.case14()
    # case_name.gen['vm_pu'] = 1.01
    p_load_ = [49, 73, 21]
    the_om, net = grid_init()
    print(net, net.res_gen)
    print('status:', net["OPF_converged"])
    # print(the_om)

