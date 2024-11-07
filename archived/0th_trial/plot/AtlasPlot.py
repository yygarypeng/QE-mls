"""
This is the class that can use python to plot the ROOT like CERN ATLAS format histograms. The know-how of using, please see the __main__ method.

Author  : Y.Y. Gary Peng
Istitute: National Tsing Hua University and ATLAS experiment, CERN
License : MIT
Data    : April 15. 2023
Email   : yuan-yen.peng@cern.ch

Version : 6.0
"""

import os
import time

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy import stats


class ATLAShist:

    """
    This is the class representing plot hep-like histograms with CERN ATLAS format.
    It should be initialized with the following parameters defined in the __init__.
    """

    def __init__(
        self,
        rmin: tuple or int,
        rmax: tuple or int,
        bins: int,
        NameOfProject: str,
        NameOfChannel: str,
        NameOfEvent: str,
    ) -> None:
        """
        This function is to initialize the the curtial parameters.

        :param rmin          : minimum of the x-axis range with default unit [GeV].
        :param rmax          : maximum of the x-axis range with default unit [GeV].
        :param bins          : number of bins.
        :param NameOfProject : Name of the project.
        :param NameofChannels: Name of the channels.
        :param NameOfEvent   : Name of the event.
        """

        self.rmin = rmin
        self.rmax = rmax
        self.nbins = bins
        self.NameOfProject = NameOfProject
        self.NameOfChannel = NameOfChannel
        self.NameOfEvent = NameOfEvent
        if type(rmin) is tuple and type(rmax) is tuple:
            self.hist_2d = True
        else:
            self.hist_2d = False

        return

    def set_data(self, data: list[np.ndarray]) -> None:
        """
        This is the function to initialize the data and set the it. The outputs are the groupped histogram, bins-array, and the input data.

        :param data: The input data.
        """
        hist_2d = self.hist_2d
        hist = []
        bins = []
        edge = []
        if hist_2d == False:
            for item in data:
                hist_temp, bins_temp = np.histogram(
                    np.array(item), bins=self.nbins, range=(self.rmin, self.rmax)
                )
                hist.append(hist_temp)
                bins.append(bins_temp)
        elif hist_2d == True:
            hist, x_edges, y_edges = np.histogram2d(
                data[0],
                data[1],
                bins=self.nbins,
                range=((self.rmin[0], self.rmax[1]), (self.rmin[0], self.rmax[1])),
                density=True,
            )
            edge = [x_edges, y_edges]

        self.data = data
        self.hist = hist
        self.bins = bins
        self.edge = edge

        return

    def set_graph(
        self,
        xlabel="x",
        ylabel="y",
        lib_name="Perliminary",
        fontsize=14.0,
        unit="GeV",
        color=["tab:red"],
    ) -> tuple:
        """
        This is a function that is to set the graph.

        :param xlabel  : Name of the graph label on x axis.
        :param ylabel  : Name of the graph label on y axis.
        :param lib_name: Name of the library on the graph, commonly used is "Perliminary", "Simulation", etc.
        :param fontsize: Size of the font on the axis label. The lib_name are default values as axis_labels + 2.
        :param units: The units, by default, are "GeV".
        :param color: The color fo the plot.
        """

        self.xl = xlabel
        self.yl = ylabel
        self.lib = lib_name
        self.size = fontsize
        self.color = color
        try:
            if type(unit) is list:
                self.unit = [unit[0], unit[1]]
                print("Use two different units...")
            elif type(unit) is str:
                self.unit = [unit, unit]
                print("Use same units...")
        except:
            print(
                "Please check the unit input, it can be a string: same unit or [string, string]: different units."
            )

        hep.style.use("ATLAS")

        return self.xl, self.yl, self.lib, self.size, self.unit, self.color

    def dump(self, filename) -> None:
        """
        This is a function to dump the results, which include figures and statistical results in .txt file.

        :param filename: What name you want to save the figure.
        """
        # figure path
        path = f"../figures/{self.NameOfProject}/{self.NameOfChannel}"
        if not os.path.isdir(path):
            os.makedirs(path)
        # save figures.
        plt.savefig(f"{path}/{filename}.png")
        print(f"... save figure: {filename}.png to ", path)
        return

    def histfill_1d(
        self,
        filename: str,
        legend_name: list[str],
        plottype="fill",
        fig_size=(9, 7),
        loc=1,
        dpi=100,
        lumi=139,
        # year=2017,
        stack=False,
        show=False,
        dump=False,
        **kargs,
    ) -> None:
        """
        This the main plotting function to plot the fill histograms; you can change to different format by changing histtype to step.

        :param filename: The name of the saved figure.
        :param legend_name: The name of the the plotted channel of interest.
        :param lumi: The number of luminosity.
        :param year: The year of the dataset.
        :param loc: The keyword arguments, here you can use "loc" with int: 0,1,2,3,4 ==> they mean different configurations of the positions of the ATLAS labels.
        :param kargs: Inherited keyword arguments from matplotlib constructor.
        :param show: If you want to show the statistical info, set it to True.
        :param others: As the text of nameings.
        """

        hist, bins = self.hist, self.bins
        width = abs(bins[0][1] - bins[0][0])
        lst_graph = [self.xl, self.yl, self.lib, self.size]
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        hep.histplot(
            hist,
            bins=bins[0],
            density=True,
            stack=stack,
            histtype=plottype,
            color=self.color,
            alpha=0.8,
            edgecolor=self.color,
            label=legend_name,
            ax=ax,
            **kargs,
        )
        ax.set_xlabel(
            lst_graph[0] + f" [{self.unit[0]}]", fontsize=lst_graph[3], labelpad=7
        )
        ax.set_ylabel(
            lst_graph[1] + f" / {width:.3}" + f" [{self.unit[0]}]",
            fontsize=lst_graph[3],
            labelpad=7,
        )

        ax.get_xaxis().get_offset_text().set_position(xy=(1.1, 1.0))
        ax.get_yaxis().get_offset_text().set_position(xy=(1.0, 0.8))

        ax.set_xlim(self.rmin, self.rmax)
        hep.atlas.label(
            lst_graph[2],
            data=True,
            lumi=lumi,
            # year=year,
            fontsize=lst_graph[3] + 2,
            loc=loc,
        )
        ax.legend(fontsize=lst_graph[3])
        fig.tight_layout()

        if dump is True:
            self.dump(filename)
        if show is True:
            i = 0
            for data in self.data:
                msg = f"""
Data {i}:
    The average of {self.NameOfEvent} is {np.average(data):>.5};
    The STD     of {self.NameOfEvent} is {stats.tstd(data):>.5};
    The SEM     of {self.NameOfEvent} is {stats.sem(data):>.5}.
"""
                print(msg)
                i += 1
        plt.show()
        plt.close()

        return print("1D, Finish!")

    def histfill_2d(
        self,
        filename: str,
        fig_size=(9, 7),
        loc=1,
        dpi=100,
        lumi=139,
        # year=2017,
        show=False,
        dump=False,
        **kargs,
    ) -> None:
        """
        This the main plotting function to plot the fill histograms; you can change to different format by changing histtype to step.

        :param filename: The name of the saved figure.
        :param legend_name: The name of the the plotted channel of interest.
        :param lumi: The number of luminosity.
        :param year: The year of the dataset.
        :param loc: The keyword arguments, here you can use "loc" with int: 0,1,2,3,4 ==> they mean different configurations of the positions of the ATLAS labels.
        :param kargs: Inherited keyword arguments from matplotlib constructor.
        :param others: As the text of nameings.
        """

        hist, edges = self.hist, self.edge
        lst_graph = [self.xl, self.yl, self.lib, self.size]
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        hep.hist2dplot(
            hist.T,
            xbins=edges[0],
            ybins=edges[1],
            cmap="viridis",
            cbar=True,
            linewidth=0,
            alpha=0.9,
            flow="both",
            ax=ax,
            **kargs,
        )
        ax.set_xlabel(
            lst_graph[0] + f" [{self.unit[0]}]", fontsize=lst_graph[3], labelpad=7
        )
        ax.set_ylabel(
            lst_graph[1] + f" [{self.unit[1]}]", fontsize=lst_graph[3], labelpad=7
        )

        ax.get_xaxis().get_offset_text().set_position(xy=(1.1, 1.0))
        ax.get_yaxis().get_offset_text().set_position(xy=(1.0, 0.8))
        ax.tick_params(axis="both", which="major", labelsize=lst_graph[3])

        ax.set_xlim(self.rmin[0], self.rmax[0])
        ax.set_ylim(self.rmin[1], self.rmax[1])

        hep.atlas.label(
            lst_graph[2],
            data=True,
            lumi=lumi,
            # year=year,
            fontsize=lst_graph[3] + 2,
            loc=loc,
        )
        fig.tight_layout()

        if dump is True:
            self.dump(filename)
        if show is True:
            plt.show()
        plt.close()

        return print("2D, Finish!")


if __name__ == "__main__":
    """
    Test the function with zz.
    ref: https://hsf-training.github.io/hsf-training-matplotlib/05-mplhep/index.html
    """

    zz = np.array(
        [
            0.181215,
            0.257161,
            0.44846,
            0.830071,
            1.80272,
            4.57354,
            13.9677,
            14.0178,
            4.10974,
            1.58934,
            0.989974,
            0.839775,
            0.887188,
            0.967021,
            1.07882,
            1.27942,
            1.36681,
            1.4333,
            1.45141,
            1.41572,
            1.51464,
            1.45026,
            1.47328,
            1.42899,
            1.38757,
            1.33561,
            1.3075,
            1.29831,
            1.31402,
            1.30672,
            1.36442,
            1.39256,
            1.43472,
            1.58321,
            1.85313,
            2.19304,
            2.95083,
        ]
    )
    ttbar = np.array(
        [
            0.00465086,
            0,
            0.00465086,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.00465086,
            0,
            0,
            0,
            0,
            0,
            0.00465086,
            0,
            0,
            0,
            0,
            0.00465086,
            0.00465086,
            0,
            0,
            0.0139526,
            0,
            0,
            0.00465086,
            0,
            0,
            0,
            0.00465086,
            0.00465086,
            0.0139526,
            0,
            0,
        ]
    )

    # single plot
    hp = ATLAShist(0, 0.015, 500, "test", "WW", "tt_test")
    hp.set_data([ttbar])
    hp.set_graph(color=["tab:green"])
    hp.histfill_1d(
        filename="test",
        legend_name=[r"$t \bar t \rightarrow 4l$"],
        show=True,
        dump=False,
    )

    # mutiple plots
    hp = ATLAShist(0, 3, 10, "test", "WW", "zzttbar_test")
    hp.set_data([zz, ttbar])
    hp.set_graph(color=["tab:red", "tab:green"])
    hp.histfill_1d(
        plottype="step",
        filename="test",
        legend_name=[r"$ZZ \rightarrow 4l$", r"$t \bar t \rightarrow 4l$"],
        show=True,
        dump=False,
        loc=0,
    )

    # 2d histogram
    x = np.random.normal(0, 10, int(1e5))
    y = np.random.normal(0, 10, int(1e5))
    # print(np.min(x), np.max(x))
    # print(np.min(y), np.max(y))
    hp = ATLAShist(
        (np.min(x), np.min(y)), (np.max(x), np.max(y)), 50, "test", "WW", "zzttbar_test"
    )
    hp.set_data([x, y])
    hp.set_graph(color=["tab:red", "tab:green"], unit=["u1", "u2"])
    hp.histfill_2d(filename="test", show=True, dump=False, loc=0)
