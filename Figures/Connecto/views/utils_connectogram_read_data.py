import math
import numpy as np
import pandas as pd


def to_cartesian(r, theta):
    """
    Converts polar r, theta (in radians) to cartesian x, y.
    """

    if theta > np.pi or theta < -np.pi:  # Converts theta (radians) to be within -pi and +pi.
        theta = theta % np.pi

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def to_polar(x, y):
    """
    Converts cartesian x, y to polar r, theta (in radians).
    """

    theta = math.atan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)

    return r, theta


def get_nodes(data_con: pd.DataFrame) -> pd.DataFrame:
    chan_ID = np.unique(np.concatenate([data_con.Stim, data_con.Chan])).astype('int')

    nodes = pd.DataFrame(chan_ID.T, columns=['ID'])

    nodes.insert(0, 'theta', 0.)
    nodes.insert(0, 'y', 0.)
    nodes.insert(0, 'x', 0.)
    nodes.insert(0, 'Label', '_')
    nodes.insert(0, 'OrderInRegion', 0)
    nodes.insert(0, 'Region', '_')
    nodes.insert(0, 'Subj', '_')

    for c in chan_ID:
        if c in data_con.Stim.values:
            nodes.loc[nodes.ID == c, ['Label', 'Region', 'Subj']] = data_con.loc[data_con.Stim == c, ['StimA', 'StimR', 'Subj']].values[0, :]
        if c in data_con.Chan.values:
            nodes.loc[nodes.ID == c, ['Label', 'Region', 'Subj']] = data_con.loc[data_con.Chan == c, ['ChanA', 'ChanR', 'Subj']].values[0, :]

    return nodes


def get_info_c(areas_c: pd.DataFrame, n_nodes: pd.DataFrame, radius: float, plot_hem: str = 'r') -> pd.DataFrame:
    for region in areas_c.region:
        areas_c.loc[areas_c.region == region, 'N_nodes'] = n_nodes[region]
    areas_c.loc[np.isnan(areas_c.N_nodes), 'N_nodes'] = np.nanmin(areas_c.N_nodes)

    ratios = areas_c.N_nodes.values
    ratios = ratios / np.min(ratios)
    ratios_n = np.ones((len(ratios),))
    ratios_n[ratios > np.percentile(ratios, 33)] = 2
    ratios_n[ratios > np.percentile(ratios, 66)] = 3

    tot_seg = np.sum(ratios_n)
    start = np.pi / 2.
    n_areas_c = len(areas_c)
    area_c_borders = np.zeros((n_areas_c, 2))
    gap = np.pi / tot_seg / 10
    for i in range(n_areas_c):
        area_c_borders[i, 0] = start - gap
        area_c_borders[i, 1] = start + gap - np.pi * ratios_n[i] / tot_seg
        start -= np.pi * ratios_n[i] / tot_seg

    areas_c.insert(5, 'theta1', area_c_borders[:, 1])
    areas_c.insert(5, 'theta0', area_c_borders[:, 0])

    areas_c_xy = np.zeros((n_areas_c, 4))
    for i in range(n_areas_c):
        areas_c_xy[i, :2] = to_cartesian(r=radius, theta=areas_c.theta0.values[i])
        areas_c_xy[i, 2:] = to_cartesian(r=radius, theta=areas_c.theta1.values[i])
    areas_c.insert(5, 'y1', areas_c_xy[:, 3])
    areas_c.insert(5, 'y0', areas_c_xy[:, 1])
    areas_c.insert(5, 'x1', areas_c_xy[:, 2])
    areas_c.insert(5, 'x0', areas_c_xy[:, 0])

    if plot_hem == 'l':
        new_x0, new_x1 = -areas_c.x0.values, -areas_c.x1.values
        areas_c.x0, areas_c.x1 = new_x0, new_x1
        for i in range(len(areas_c)):
            _, areas_c.theta1.values[i] = to_polar(areas_c.x0.values[i], areas_c.y0.values[i])
            _, areas_c.theta0.values[i] = to_polar(areas_c.x1.values[i], areas_c.y1.values[i])
    return areas_c


def get_info_s(areas_s: pd.DataFrame, n_nodes: pd.DataFrame, radius: float, plot_hem: str = 'r') -> (pd.DataFrame, float):
    n_areas_s = len(areas_s)
    areas_s = areas_s.sort_values(by=['plot_order'])
    areas_s = areas_s.reset_index(drop=True)

    # node ratio to get segment size
    for region in areas_s.region:
        if region in n_nodes:
            areas_s.loc[areas_s.region == region, 'N_nodes'] = n_nodes[region]
        else:
            areas_s.loc[areas_s.region == region, 'N_nodes'] = 1
    areas_s.loc[np.isnan(areas_s.N_nodes), 'N_nodes'] = np.nanmin(areas_s.N_nodes)

    ratios = areas_s.N_nodes.values
    ratios = ratios / np.min(ratios)
    ratios_n = np.ones((len(ratios),))
    ratios_n[ratios > np.percentile(ratios, 33)] = 2
    ratios_n[ratios > np.percentile(ratios, 66)] = 3

    y_start = int(radius - 0.1 * radius)
    y_end = int(-radius + 0.1 * radius)
    total_length = y_start - y_end
    tot_seg = int(np.sum(ratios_n))
    areas_s_xy = np.zeros((n_areas_s, 2, 2))
    areas_s_xy[:, :, 0] = -0.2 * radius if plot_hem == 'r' else +0.2 * radius
    start = y_start
    gap = 0.1 * total_length / tot_seg
    for i in range(n_areas_s):
        areas_s_xy[i, 0, 1] = start - gap
        areas_s_xy[i, 1, 1] = start + gap - total_length * ratios_n[i] / tot_seg
        start -= total_length * ratios_n[i] / tot_seg
    areas_s.insert(5, 'y1', areas_s_xy[:, 1, 1])
    areas_s.insert(5, 'y0', areas_s_xy[:, 0, 1])
    areas_s.insert(5, 'x1', areas_s_xy[:, 1, 0])
    areas_s.insert(5, 'x0', areas_s_xy[:, 0, 0])

    y_lin = np.linspace(y_start, y_end, tot_seg + 1)
    l_s = 0.9 * abs(y_lin[0] - y_lin[1])

    return areas_s, l_s


def get_nodes_coords(nodes: pd.DataFrame, areas_c: pd.DataFrame, areas_s: pd.DataFrame, r_nodes: float) -> pd.DataFrame:
    for region in areas_c.region:
        n_node = len(nodes[nodes.Region == region])
        if n_node > 0:
            t = areas_c.loc[areas_c.region == region, ['theta0', 'theta1']].values[0]
            d0 = 2. * np.pi if (t[0] < 0) & (t[1] > 0) else 0
            theta = (t[0] + t[1] + d0) / 2. if n_node == 1 else np.linspace(t[1], t[0] + d0, n_node)  # Place the single node in the middle of the range if only one node
            theta = np.where(theta > np.pi, theta - 2. * np.pi, theta)
            nodes.loc[nodes.Region == region, 'theta'] = theta

            xy = np.zeros((n_node, 2))
            for i in range(n_node):
                x, y = to_cartesian(r=r_nodes, theta=nodes.loc[nodes.Region == region, 'theta'].values[i])
                xy[i, 0] = x
                xy[i, 1] = y
            nodes.loc[nodes.Region == region, 'x'] = xy[:, 0]
            nodes.loc[nodes.Region == region, 'y'] = xy[:, 1]

    for region in areas_s.region:
        n_node = len(nodes[nodes.Region == region])
        if n_node > 0:
            y = areas_s.loc[areas_s.region == region, ['y0', 'y1']].values[0]
            x_base = areas_s.loc[areas_s.region == region, ['x0']].values[0][0]

            xy = np.zeros((n_node, 2))
            xy[:, 0] = x_base - np.sign(x_base) * 2.

            if n_node == 1:  # put in the middle if only one node
                xy[0, 1] = (y[0] + y[1]) / 2
            else:
                xy[:, 1] = np.linspace(y[0], y[1], n_node)

            nodes.loc[nodes.Region == region, 'x'] = xy[:, 0]
            nodes.loc[nodes.Region == region, 'y'] = xy[:, 1]

    return nodes
