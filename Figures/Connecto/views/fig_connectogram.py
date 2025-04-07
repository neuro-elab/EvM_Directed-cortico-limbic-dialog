import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.figure import Figure
import matplotlib.lines as mlines

from source.globals import ATLAS, PLOT_ATLAS, REGION_ABBREVIATION
import views.utils_connectogram_read_data as rd
from views.utils_connectogram_axes import plot_curved_text, despine, initialize_data_edges, get_line_style_and_width, \
    get_colors, P

plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10,
})


class Connectogram(Figure):
    def __init__(self, ax=None) -> None:
        super().__init__()
        self.ax = ax  # Use the provided ax for plotting
        self.ax_nodes = None

        self.data_con = None
        self.data_nodes = None
        self.data_edges = None
        self.plot_hem = None

        self.r_seg = 20
        self.radius = 20
        self.r_nodes = 18

        self.ylim = [-10, 22]
        self.xlim = [-25, 25]
        self.linesIn = []
        self.linesOut = []

    def setData(self, data_con: pd.DataFrame, data_nodes: pd.DataFrame, plot_hem: str = 'r'):
        self.data_con = data_con
        self.data_nodes = data_nodes
        self.data_nodes = self.data_nodes.sort_values(by=['Label', 'Region', 'Subj'])
        self.data_nodes = self.data_nodes.reset_index(drop=True)

        self.plot_hem = plot_hem
        self.ylim = [-22, 10] if self.plot_hem == 'l' else [-10, 22]
        self.linesIn = []
        self.linesOut = []

        self._load_information()

    def _load_information(self):
        areas_c = PLOT_ATLAS[PLOT_ATLAS.plot_pos == 'c']
        areas_c.insert(5, 'N_nodes', np.nan)
        areas_c = areas_c.sort_values(by=['plot_order']).reset_index(drop=True)
        areas_s = PLOT_ATLAS[PLOT_ATLAS.plot_pos == 's']

        self.data_nodes = self.data_nodes.reset_index(drop=True)
        for subregion in np.unique(self.data_nodes.Label):
            self.data_nodes.loc[self.data_nodes.Label == subregion, 'Label'] = subregion.replace(" ", "")
            subregion = subregion.replace(" ", "")
            region = ATLAS.loc[ATLAS.abbreviation == subregion, 'region'].values
            order_in_region = ATLAS.loc[ATLAS.abbreviation == subregion, 'anatomical_order_in_region'].values
            if len(region) > 0:
                self.data_nodes.loc[self.data_nodes.Label == subregion, 'Region'] = region[0]
                self.data_nodes.loc[self.data_nodes.Label == subregion, 'OrderInRegion'] = order_in_region[0]
            #else: Just to test which labels are not in a given region
            #    print(subregion)

        self.data_nodes = self.data_nodes.sort_values(by=['Region', 'OrderInRegion'])

        # add dummy subregion to have all regions plotted
        ID_dummy = np.max(self.data_nodes.ID) + 1
        regions = np.unique(PLOT_ATLAS.region)
        for region in regions:
            if region not in np.unique(self.data_nodes.Region):
                subregion_dummy = ATLAS.loc[ATLAS.region == region, 'abbreviation'].values[0]
                pd_dummy = pd.DataFrame([['dummy', region, 0, subregion_dummy, ID_dummy, 0, 0, 0]],
                                        columns=self.data_nodes.columns)
                self.data_nodes = pd.concat([self.data_nodes, pd_dummy])
                ID_dummy += 1

        n_nodes = self.data_nodes.groupby(['Region'])['ID'].count()
        self.areas_s, self.l_s = rd.get_info_s(areas_s, n_nodes, self.r_seg, self.plot_hem)
        self.areas_c = rd.get_info_c(areas_c, n_nodes, self.r_seg, self.plot_hem)
        self.data_nodes = rd.get_nodes_coords(self.data_nodes, self.areas_c, self.areas_s, self.r_nodes)

    def plot_nodes(self, show_label: bool = True):
        if self.ax is None:
            raise ValueError("No axes instance provided.")
        ax = self.ax  # Use the class's ax if none provided

        outer_width = 0.05 * self.r_seg
        inner_width = 0.05 * self.r_seg

        for i in range(len(self.areas_c)):
            t = self.areas_c[['theta0', 'theta1']].values[i]

            outer_ring = mpatches.Wedge((0, 0), self.r_seg + outer_width, math.degrees(t[1]) + 90,
                                        math.degrees(t[0]) + 90,
                                        width=outer_width * 2, color=self.areas_c.plot_color.values[i])
            ax.add_patch(outer_ring)

            inner_ring = mpatches.Wedge((0, 0), self.r_seg, math.degrees(t[1]) + 90, math.degrees(t[0]) + 90,
                                        width=inner_width, color=self.areas_c.plot_color.values[i], alpha=0.1)
            ax.add_patch(inner_ring)

            if show_label:
                abbr = REGION_ABBREVIATION.get(self.areas_c.region.values[i], 'Unknown')
                plot_curved_text(ax, self.r_seg, t[0], t[1], abbr)

        for i in range(len(self.areas_s)):
            l_s = self.areas_s.y1.values[i] - self.areas_s.y0.values[i]

            rectangle = mpatches.Rectangle((-self.areas_s.y0.values[i], self.areas_s.x0.values[i]),
                                           -l_s, (-1) ** (np.array(self.plot_hem == 'r')) * inner_width * 2,
                                           color=self.areas_s.plot_color.values[i])
            ax.add_patch(rectangle)

            x_subregion = self.areas_s.x0.values[i] - (-1) ** (np.array(self.plot_hem == 'r')) * inner_width
            rectangle = mpatches.Rectangle((-self.areas_s.y0.values[i], x_subregion),
                                           -l_s, (-1) ** (np.array(self.plot_hem == 'r')) * inner_width,
                                           color=self.areas_s.plot_color.values[i], alpha=0.1)
            ax.add_patch(rectangle)

            x = self.areas_s.x0.values[i] - 2 * 0.03 * self.r_seg * (-1) ** (np.array(self.plot_hem == 'l') * 1)
            y = self.areas_s.y1.values[i] + (self.areas_s.y0.values[i] - self.areas_s.y1.values[i]) / 2.

            if show_label:
                abbr = REGION_ABBREVIATION.get(self.areas_s.region.values[i], 'Unknown')
                ax.text(s=abbr + '.', x=-y, y=x, ha='center', va='center', fontsize=7, color='white')

        # Nodes
        for i in range(len(self.data_nodes)):
            node_patch = mpatches.Circle((-self.data_nodes.y.values[i], self.data_nodes.x.values[i]), 0.1, lw=1, zorder=2)
            ax.add_patch(node_patch)

        shift = 0  # 0.08
        x = self.areas_s.x0.values[-1] - (-1) ** (np.array(self.plot_hem == 'r')) * inner_width
        for area in np.unique(self.data_nodes.Label):
            nodes_area = self.data_nodes[self.data_nodes.Label == area]
            region = nodes_area.Region.values[0]

            if np.isin(region, self.areas_s.region):  # straight
                col = self.areas_s.plot_color.values[self.areas_s.region == region][0]
                y0 = nodes_area.y.values[0]
                y1 = nodes_area.y.values[-1]

                rectangle = mpatches.Rectangle(xy=(-y0 + shift, x),
                                               width=-((y1 - y0) + 2 * shift),
                                               height=(-1) ** (np.array(self.plot_hem == 'r')) * inner_width,
                                               color=col, alpha=np.random.random() / 3. + 0.5)
                ax.add_patch(rectangle)
            else:
                col = self.areas_c.plot_color.values[self.areas_c.region == region][0]
                t = [nodes_area.theta.values[0], nodes_area.theta.values[-1]]
                ring = mpatches.Wedge(center=(0, 0), r=0.95 * self.r_seg,
                                      theta1=math.degrees(np.min(t)) - 0.1 + 90,
                                      theta2=math.degrees(np.max(t)) + 0.1 + 90,
                                      width=0.03 * self.r_seg, color=col, alpha=np.random.random() / 3 + 0.5)
                ax.add_patch(ring)

            self.ax_nodes = ax

            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            despine(ax)

    def plot_with_dashes(self, ax, x, y, color, alpha, linewidth, linestyle, dashes):
        line = mlines.Line2D(x, y, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        if dashes:
            line.set_dashes(dashes)
        ax.add_line(line)
        return [line]

    def show_con_all(self, edges: pd.DataFrame, color_style: str = 'area', lw_max: float = 2, bi_dir: bool = False,
                     arrow_end: bool = False, cmap: str = 'hot', vmin: float = None, vmax: float = None):
        colormap = plt.get_cmap(cmap) if cmap is not None else plt.get_cmap('hot')

        edges = initialize_data_edges(edges, bi_dir)
        edges['lw'] = edges['lw'] * lw_max
        edges['lw_out'] = edges['lw_out'] * lw_max
        self.data_edges = edges
        ax = self.ax

        # Edges
        n = 100
        n_half = int(n / 2.)
        tt = np.linspace(0, 1, n)
        for i in range(len(self.data_edges)):
            c0 = self.data_edges.Stim.values[i]
            c1 = self.data_edges.Chan.values[i]
            if (len(self.data_nodes[self.data_nodes.ID == c1]) > 0) \
                    & (len(self.data_nodes[self.data_nodes.ID == c0]) > 0):
                r1 = self.data_nodes.loc[self.data_nodes.ID == c0, 'Region'].values[0]
                xy0 = self.data_nodes.loc[self.data_nodes.ID == c0, ['x', 'y']].values[0]
                xy1 = self.data_nodes.loc[self.data_nodes.ID == c1, ['x', 'y']].values[0]
                verts = [xy0,
                         ((-1) ** (np.array(self.plot_hem == 'l') * 1) * self.r_nodes / 3., 0.),
                         xy1]
                xx = P(tt, np.array(verts))

                # get color and linestyle
                lw, ls, lw_out, ls_out = get_line_style_and_width(edges, i, bi_dir)
                col, col_out = get_colors(color_style, edges, i, r1, vmin, vmax, colormap, bi_dir)
                # Set dashes only if linestyle is '--'
                dashes = (2/lw, 1/lw) if ls == '--' else None
                dashes_out = (2/lw_out, 1/lw_out) if ls_out == '--' else None
                if bi_dir:  # Show different color in one line for two directional effect
                    lineOut = self.plot_with_dashes(ax, -xx[:n_half, 1], xx[:n_half, 0], col_out, 0.7, lw_out, ls_out, dashes_out)
                    lineIn = self.plot_with_dashes(ax, -xx[n_half - 1:, 1], xx[n_half - 1:, 0], col, 0.7, lw, ls, dashes)
                    self.linesOut.append(lineOut[0])
                elif color_style == 'area':
                    lineOut = self.plot_with_dashes(ax, -xx[:n_half, 1], xx[:n_half, 0], col_out, 0.7, lw_out, ls_out, dashes_out)
                    lineIn = self.plot_with_dashes(ax, -xx[n_half - 1:, 1], xx[n_half - 1:, 0], col, 0.7, lw, ls, dashes)
                    self.linesOut.append(lineOut[0])
                else:
                    lineIn = self.plot_with_dashes(ax, -xx[:, 1], xx[:, 0], col, 0.7, lw, ls, dashes)
                self.linesIn.append(lineIn[0])
                # if bi_dir:  # show different color in one line for two directional effect
                #     lineOut = # ax.plot(-xx[:n_half, 1], xx[:n_half, 0], color=col_out, alpha=0.7, linewidth=lw_out,
                #                       ls=ls_out, dahses = dashes)
                #     lineIn = ax.plot(-xx[n_half - 1:, 1], xx[n_half - 1:, 0], color=col, alpha=0.7, linewidth=lw,
                #                      ls=ls, dashes = dashes)
                #     self.linesOut.append(lineOut[0])
                # elif color_style == 'area':
                #     lineOut = ax.plot(-xx[:n_half, 1], xx[:n_half, 0], color=col_out, alpha=0.7, linewidth=lw_out,
                #                       ls=ls_out, dashes = dashes)
                #     lineIn = ax.plot(-xx[n_half - 1:, 1], xx[n_half - 1:, 0], color=col, alpha=0.7, linewidth=lw,
                #                      ls=ls, dashes = dashes)
                #     self.linesOut.append(lineOut[0])
                # else:
                #     lineIn = ax.plot(-xx[:, 1], xx[:, 0], color=col, alpha=0.7, linewidth=lw, dashes = dashes)
                # self.linesIn.append(lineIn[0])

                # Arrow head
                if arrow_end:
                    frac_len = int(0.05 * len(xx))
                    frac_len = max(frac_len, 1)  # ensure that it is at least 1 to avoid indexing issues
                    arrow = FancyArrowPatch(
                        (-xx[-frac_len - 1, 1], xx[-frac_len - 1, 0]),  # Start point
                        (-xx[-1, 1], xx[-1, 0]),  # End point
                        mutation_scale=10,
                        color=col,
                        linewidth=lw,
                        arrowstyle='-|>'
                    )
                    ax.add_patch(arrow)

                    if bi_dir | (color_style == 'area'):
                        # Add an arrowhead
                        frac_len = int(0.05 * len(xx))
                        frac_len = max(frac_len, 1)  # ensure that it is at least 1 to avoid indexing issues
                        arrow = FancyArrowPatch(
                            (-xx[0, 1], xx[0, 0]),  # Start point
                            (-xx[frac_len, 1], xx[frac_len, 0]),  # End point
                            mutation_scale=10,
                            color=col_out,
                            linewidth=lw_out,
                            arrowstyle='<|-'
                        )
                        ax.add_patch(arrow)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        despine(ax)
