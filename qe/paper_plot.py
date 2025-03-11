import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

hep.style.use("ATLAS")


class Plotter:
    def __init__(self):
        pass

    ########################
    # 1D plot
    ########################
    def hist_1d(
        self,
        true,
        pred,
        ranges=[0.0, 1.0],
        xlabel=r"$p_{z}^{\nu\nu}$ [GeV]",
        title="",
        ylabel="Counts",
        legend=["True", "Pred"],
        bins=50,
        xpad=10,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 1D histogram comparing true and predicted values.

        Parameters:
        :param true: array-like, true values
        :param pred: array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str, label for the x-axis
        :param title: str, title of the plot
        :param ylabel: str, label for the y-axis
        :param sub_ylabel: str, label for the subplot y-axis
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [6, 2], "hspace": 0.1},
            sharex=True,
        )

        truth_bar, truth_bin = np.histogram(
            true, bins=bins, range=ranges, weights=weights
        )
        pred_bar, _ = np.histogram(pred, bins=bins, range=ranges, weights=weights)

        hep.histplot(truth_bar, truth_bin, label=f"{legend[0]}", ax=ax1, lw=2, color="b")
        hep.histplot(pred_bar, truth_bin, label=f"{legend[1]}", ax=ax1, lw=2, color="r")

        label_size = 24
        tick_size = 22
        title_size = 20

        ax2.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax1.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax2.xaxis.offsetText.set_fontsize(tick_size)
        ax1.yaxis.offsetText.set_fontsize(tick_size)

        ax1.set_xlim(ranges)
        ax1.legend(fontsize=tick_size)
        ax1.set_ylabel(ylabel, fontsize=label_size, labelpad=5)
        ax1.set_title(title, fontsize=title_size, loc="right")

        ratio = np.divide(pred_bar + 1, truth_bar + 1, where=(truth_bar != 0))
        ax2.vlines(truth_bin[1:], 1, ratio, color="k", lw=1)

        for i, val in enumerate(ratio):
            if val >= 2:
                ax2.annotate(
                    "",
                    xy=(truth_bin[i + 1], 2),
                    xytext=(truth_bin[i + 1], 1.95),
                    arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                )
            elif val <= 0:
                ax2.annotate(
                    "",
                    xy=(truth_bin[i + 1], 0),
                    xytext=(truth_bin[i + 1], 0.1),
                    arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                )
            else:
                ax2.scatter(truth_bin[i + 1], val, color="k", lw=1, s=10)

        ax2.set_ylim([0, 2])
        ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)

        ax2.set_xlabel(xlabel, fontsize=label_size, labelpad=xpad)
        ax2.set_ylabel(f"{legend[1]}/{legend[0]}", fontsize=label_size, loc="center", labelpad=10)

        ax1.tick_params(axis="y", labelsize=tick_size)
        ax2.tick_params(axis="x", labelsize=tick_size, pad=10)
        ax2.tick_params(axis="y", labelsize=tick_size)

        if save_name is not None:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 1D plot as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # 2D plot
    ########################
    def hist_2d(
        self,
        true,
        pred,
        ranges=[0.0, 1.0],
        xlabel=r"True $p_{z}^{\nu\nu}$ [GeV]",
        ylabel=r"Predicted $p_{z}^{\nu\nu}$ [GeV]",
        title=r"",
        bins=50,
        xpad=10,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2D histogram comparing true and predicted values.

        Parameters:
        :param true: array-like, true values
        :param pred: array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str, label for the x-axis
        :param ylabel: str, label for the y-axis
        :param title: str, title of the plot
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """
        if ranges is None:
            ranges = [
                np.min([np.min(pred), np.min(true)]),
                np.max([np.max(pred), np.max(true)]),
            ]

        label_size = 24
        tick_size = 22
        title_size = 20

        fig, ax = plt.subplots(figsize=(8, 8))  # Make the figure square
        h = ax.hist2d(
            pred,
            true,
            bins=bins,
            range=[ranges, ranges],
            cmap="viridis",
            cmin=1,
            norm=LogNorm(),
            weights=weights,
        )
        cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("", fontsize=tick_size)
        cbar.ax.tick_params(axis="both", which="both", labelsize=tick_size)
        ax.set_title(title, fontsize=title_size, loc="right")
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=xpad)
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=5)
        ax.plot(ranges, ranges, color="grey", linestyle="--", alpha=0.8)  # add y=x line
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.xaxis.offsetText.set_fontsize(tick_size)
        ax.yaxis.offsetText.set_fontsize(tick_size)
        ax.tick_params(axis="both", labelsize=tick_size, pad=10)
        ax.set_xlim(ranges)
        ax.set_ylim(ranges)

        if save_name is not None:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 2D plot as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # 1D Grid (2×3)
    ########################
    def hist_1d_grid(
        self,
        true_list,
        pred_list,
        ranges=(0.0, 1.0),
        xlabel="[unit]",
        title="",
        ylabel="Counts",
        legend_lst=["Pred", "True"],
        bins=50,
        xpad=8,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2*3 grid of 1D histograms comparing true and predicted values.

        :param true_list: list of array-like, true values
        :param pred_list: list of array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str or list, label for the x-axis
        :param title: str or list, title of the plot
        :param ylabel: str or list, label for the y-axis
        :param sub_ylabel: str or list, label for the subplot y-axis
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """

        # Expand single-value params into lists
        if not isinstance(ranges, list):
            ranges = [ranges] * 6
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 6
        if not isinstance(title, list):
            title = [title] * 6
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * 6
        sub_ylabel = f"{legend_lst[0]}/{legend_lst[1]}"
        if not isinstance(sub_ylabel, list):
            sub_ylabel = [sub_ylabel] * 6

        label_size = 18
        tick_size = 14
        title_size = 18

        fig = plt.figure(figsize=(20, 14))
        outer = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.08)

        for i in range(6):
            # Determine row & col in the 2×3 grid
            row = i // 3  # 0 for top row, 1 for bottom row
            col = i % 3  # 0,1,2 from left to right

            # Nested GridSpec for main (ax1) & ratio (ax2)
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[i], height_ratios=[6, 2], hspace=0.08
            )
            ax1 = fig.add_subplot(inner[0])
            ax2 = fig.add_subplot(inner[1], sharex=ax1)

            # Remove the top subplot's x tick labels (ratio subplot handles x)
            ax1.tick_params(labelbottom=False)

            # Hist data
            tr_bar, tr_bin = np.histogram(
                true_list[i], bins=bins, range=ranges[i], weights=weights
            )
            pr_bar, _ = np.histogram(
                pred_list[i], bins=bins, range=ranges[i], weights=weights
            )

            # Plot main hist
            hep.histplot(tr_bar, tr_bin, ax=ax1, lw=2, color="b", label=legend_lst[1])
            hep.histplot(pr_bar, tr_bin, ax=ax1, lw=2, color="r", label=legend_lst[0])
            ax1.set_xlim(ranges[i])

            # Example: RMSE & SEM in the title
            diff = np.array(pred_list[i]) - np.array(true_list[i])
            rmse = np.sqrt(np.mean(diff**2))
            sem = stats.sem(diff)
            ax1.set_title(
                f"{title[i]} (RMSE={rmse:.2f} ± {sem:.2f})",
                fontsize=title_size,
                loc="right",
            )
            ax1.legend(fontsize=tick_size)

            # Ratio logic with arrow snippet
            ratio = np.divide(pr_bar + 1, tr_bar + 1, where=(tr_bar != 0))
            ax2.vlines(tr_bin[1:], 1, ratio, color="k", lw=1)
            for j, val in enumerate(ratio):
                if val > 2:
                    # plot the arrowon 2 which force the val to be 1.9 and plot the arrow
                    ax2.annotate(
                        "",
                        xy=(tr_bin[j + 1], 2),
                        xytext=(tr_bin[j + 1], 1.95),
                        arrowprops=dict(
                            facecolor="k", shrink=0.05, width=1, headwidth=2
                        ),
                    )
                elif val < 0:
                    # do the similar trick!
                    ax2.annotate(
                        "",
                        xy=(tr_bin[j + 1], 0),
                        xytext=(tr_bin[j + 1], 0.05),
                        arrowprops=dict(
                            facecolor="k", shrink=0.05, width=1, headwidth=2
                        ),
                    )
                else:
                    ax2.scatter(tr_bin[j + 1], val, color="k", lw=1, s=10)

            ax2.set_ylim([0, 2])
            ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)

            # ========== Keep y-labels only for left column =============
            if col == 0:
                # Keep the main hist's y-label
                ax1.set_ylabel(ylabel[i], fontsize=label_size)
                # Keep the ratio subplot's y-label
                ax2.set_ylabel(
                    sub_ylabel[i], fontsize=label_size, loc="center", labelpad=10
                )
            else:
                # Remove y-axis labels
                ax1.set_ylabel("")
                ax2.set_ylabel("")
                ax1.tick_params(labelleft=False)
                ax2.tick_params(labelleft=False)

            # ========== Keep x-labels only for bottom row ==============
            if row == 1:
                ax2.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
            else:
                ax2.set_xlabel("")
                ax2.tick_params(labelbottom=False)

            # Ticks
            ax1.tick_params(axis="both", labelsize=tick_size)
            ax2.tick_params(axis="both", labelsize=tick_size, pad=10)

        # Shrink the gap between the 6 plots
        plt.subplots_adjust(
            left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.1, hspace=0.05
        )

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 1D grid as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # 2D Grid (2×3)
    ########################
    def hist_2d_grid(
        self,
        true_list,
        pred_list,
        ranges=(0.0, 1.0),
        xlabel="X",
        ylabel="Y",
        title="",
        bins=50,
        xpad=15,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2*3 grid of 2D histograms comparing true and predicted values.

        :param true_list: list of array-like, true values
        :param pred_list: list of array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str or list, label for the x-axis
        :param ylabel: str or list, label for the y-axis
        :param title: str or list, title of the plot
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """
        # Expand single-value params
        if not isinstance(ranges, list):
            ranges = [ranges] * 6
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 6
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * 6
        if not isinstance(title, list):
            title = [title] * 6

        label_size = 18
        tick_size = 14
        title_size = 18

        fig = plt.figure(figsize=(20, 14))
        # 2 rows x 4 columns (last col for colorbar)
        gs = gridspec.GridSpec(
            2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12, hspace=0.005
        )

        axes = []
        for i in range(6):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

        # Find global max/min counts for consistent color scaling
        max_count = 0
        min_count = float('inf')
        
        # Create temporary histograms to find global min/max values
        for i in range(min(len(true_list), len(pred_list), 6)):
            hist_vals, _, _ = np.histogram2d(
                pred_list[i],
                true_list[i],
                bins=bins,
                range=[ranges[i], ranges[i]],
                weights=weights
            )
            max_count = max(max_count, np.max(hist_vals))
            # Find minimum non-zero value for LogNorm
            non_zero_min = np.min(hist_vals[hist_vals > 0]) if np.any(hist_vals > 0) else 1
            min_count = min(min_count, non_zero_min)
        
        # Create a shared normalization for all plots
        norm = LogNorm(vmin=min_count, vmax=max_count)

        # Now create the actual plots with shared normalization
        for i, ax in enumerate(axes):
            row = i // 3
            col = i % 3

            h = ax.hist2d(
                pred_list[i],
                true_list[i],
                bins=bins,
                range=[ranges[i], ranges[i]],
                cmap="viridis",
                cmin=1,
                norm=norm,  # Use the shared normalization
                weights=weights,
            )

            # Diagonal
            ax.plot(ranges[i], ranges[i], "k--", alpha=0.7)

            # R2
            r2_val = r2_score(true_list[i], pred_list[i])
            ax.set_title(
                rf"{title[i]} ($R^2$={r2_val:.2f})", fontsize=title_size, loc="right"
            )

            # Only left col has y-label
            if col == 0:
                ax.set_ylabel(ylabel[i], fontsize=label_size)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            # Only bottom row has x-label
            if row == 1:
                ax.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
                ax.tick_params(axis="both", labelsize=tick_size, pad=10)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(ranges[i])
            ax.set_ylim(ranges[i])
            ax.tick_params(axis="both", labelsize=tick_size)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))

        # Create a ScalarMappable with the same normalization for the colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])  # Empty array for the mappable
        
        # Colorbar on last column that spans both rows
        cax = fig.add_subplot(gs[:, 3])  # The [:, 3] spans both rows
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.ax.text(
            2.7,
            0.5,  # Center position
            "",
            va="center",  # Vertical center alignment
            ha="left",
            rotation=-90,
            fontsize=label_size,
        )
        cbar.ax.tick_params(labelsize=tick_size)

        # Shrink gap
        plt.subplots_adjust(
            left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.12, hspace=0.01
        )

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 2D grid as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # Combined Grid (2×3)
    ########################

    def hist_combined_grid(
        self,
        true_list_1d,
        pred_list_1d,
        true_list_2d,
        pred_list_2d,
        ranges=(-1, 1),
        xlabel="X",
        title="",
        row1_ylabel="Counts",
        row1_legend=["Seperable", "SM"],
        row2_xlabel="True",
        row2_ylabel="Predicted",
        bins=50,
        xpad=1,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2x3 grid where first row has 1D histograms and second row has 2D histograms.
        """
        # Expand single-value parameters to lists if needed
        if not isinstance(ranges, list):
            ranges = [ranges] * 3
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 3
        if not isinstance(title, list):
            title = [title] * 3
            
        label_size = 18
        tick_size = 14
        title_size = 18
        
        # Create figure with GridSpec layout - including column for colorbar
        fig = plt.figure(figsize=(18, 14))
        
        # Main grid: 2 rows, 4 columns (3 for plots + 1 narrow for colorbar)
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15, hspace=0.1)
        
        # First row: 1D histograms with ratio plots (spans only first 3 columns)
        first_row_axes = []
        for col in range(3):
            # Each 1D plot consists of main plot and ratio subplot
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[0, col], height_ratios=[6, 2], hspace=0.08
            )
            ax1 = fig.add_subplot(inner_gs[0])  # Main histogram
            ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)  # Ratio plot
            first_row_axes.append((ax1, ax2))
            
            # Hide x-tick labels on main plot
            ax1.tick_params(labelbottom=False)
        
        # Second row: 2D histograms (also spans only first 3 columns)
        second_row_axes = []
        for col in range(3):
            ax = fig.add_subplot(gs[1, col])
            second_row_axes.append(ax)
        
        # Find global max count for consistent color scaling across all 2D histograms
        max_count = 0
        min_count = float('inf')
        
        # Create temporary histograms to find max/min counts
        for i in range(min(len(true_list_2d), len(pred_list_2d), 3)):
            hist_vals, _, _ = np.histogram2d(
                pred_list_2d[i],
                true_list_2d[i],
                bins=bins,
                range=[ranges[i], ranges[i]],
                weights=weights
            )
            max_count = max(max_count, np.max(hist_vals))
            # Find minimum non-zero value for LogNorm
            non_zero_min = np.min(hist_vals[hist_vals > 0]) if np.any(hist_vals > 0) else 1
            min_count = min(min_count, non_zero_min)
        
        # Create a shared normalization for all 2D plots
        norm = LogNorm(vmin=min_count, vmax=max_count)
        
        # Generate first row (1D histograms with ratios)
        for i, (ax1, ax2) in enumerate(first_row_axes):
            # Skip if beyond data range
            if i >= len(true_list_1d) or i >= len(pred_list_1d):
                continue

            # Histogram data
            tr_bar, tr_bin = np.histogram(
                true_list_1d[i], bins=bins, range=ranges[i], weights=weights
            )
            pr_bar, _ = np.histogram(
                pred_list_1d[i], bins=bins, range=ranges[i], weights=weights
            )

            # Plot main histogram
            hep.histplot(tr_bar, tr_bin, ax=ax1, lw=2, color="b", label=row1_legend[1])
            hep.histplot(pr_bar, tr_bin, ax=ax1, lw=2, color="r", label=row1_legend[0])
            ax1.set_xlim(ranges[i])
            
            # Add statistical metrics to title
            diff = np.array(pred_list_1d[i]) - np.array(true_list_1d[i])
            rmse = np.sqrt(np.mean(diff**2))
            sem = stats.sem(diff)
            ax1.set_title(
                f"{title[i]} (RMSE={rmse:.2f} ± {sem:.2f})",
                fontsize=title_size, 
                loc="right"
            )
            ax1.legend(fontsize=tick_size)
            
            # Only leftmost column gets y-label
            if i == 0:
                ax1.set_ylabel(row1_ylabel, fontsize=label_size)
            else:
                ax1.set_ylabel("")
                ax1.tick_params(labelleft=False)
            
            # Ratio plot
            ratio = np.divide(pr_bar + 1, tr_bar + 1, where=(tr_bar != 0))
            ax2.vlines(tr_bin[1:], 1, ratio, color="k", lw=1)
            
            # Handle out-of-range ratio values
            for j, val in enumerate(ratio):
                if val > 2:
                    ax2.annotate(
                        "", 
                        xy=(tr_bin[j + 1], 2),
                        xytext=(tr_bin[j + 1], 1.95),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2)
                    )
                elif val < 0:
                    ax2.annotate(
                        "",
                        xy=(tr_bin[j + 1], 0),
                        xytext=(tr_bin[j + 1], 0.05),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2)
                    )
                else:
                    ax2.scatter(tr_bin[j + 1], val, color="k", lw=1, s=10)
            
            ax2.set_ylim([0, 2])
            ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)
            ax2.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
            
            # Only leftmost column gets y-label
            if i == 0:
                ax2.set_ylabel(f"{row1_legend[0]}/{row1_legend[1]}", fontsize=label_size, loc="center", labelpad=10)
            else:
                ax2.set_ylabel("")
                ax2.tick_params(labelleft=False)
                
            # Format ticks
            ax1.tick_params(axis="both", labelsize=tick_size)
            ax2.tick_params(axis="both", labelsize=tick_size, pad=10)
        
        # Generate second row (2D histograms)
        for i, ax in enumerate(second_row_axes):
            # Skip if beyond data range
            if i >= len(true_list_2d) or i >= len(pred_list_2d):
                continue
                
            h = ax.hist2d(
                pred_list_2d[i],
                true_list_2d[i],
                bins=bins,
                range=[ranges[i], ranges[i]],
                cmap="viridis",
                cmin=1,
                norm=norm,  # Use the shared normalization
                weights=weights
            )
            
            # Add diagonal line
            ax.plot(ranges[i], ranges[i], "k--", alpha=0.7)
            
            # Add R² score to title
            r2_val = r2_score(true_list_2d[i], pred_list_2d[i])
            ax.set_title(
                f"{title[i]} ($R^2$={r2_val:.2f})",
                fontsize=title_size,
                loc="right"
            )
            
            # Format axes
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(ranges[i])
            ax.set_ylim(ranges[i])
            ax.tick_params(axis="both", labelsize=tick_size)
            
            # Only leftmost gets y-label
            if i == 0:
                ax.set_ylabel(row2_ylabel, fontsize=label_size)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
                
            ax.set_xlabel(row2_xlabel, fontsize=label_size, labelpad=xpad)
            
            # Format number display
            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))

        # Create colorbar for the 2D plots using the dedicated colorbar column
        # We need a dummy mappable with the correct normalization
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])  # Empty array for the mappable
        
        # Add colorbar in the 4th column, but only for the second row
        # cax = fig.add_subplot(gs[1, 3])

        # After plotting, finalize layout
        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.15, hspace=0.1)

        # Get the position of the rightmost plot in the second row
        last_ax_pos = second_row_axes[2].get_position()

        # Add a new axis for the colorbar that matches the height of the last plot
        cax = fig.add_axes([
            last_ax_pos.x1 + 0.01,  # right of the last plot
            last_ax_pos.y0,         # align bottoms
            0.013,                  # fixed width
            last_ax_pos.height      # same height
        ])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=tick_size)
        
        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved combined grid as {save_name}.png")
        
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Example usage
    plot = Plotter()

    # test individual plots
    pred = np.random.normal(0, 1, 10000)
    truth = np.random.rand(10000)
    plot.hist_1d(
        pred,
        truth,
        ranges=(0, 10000),
        xpad=25,
        save_name="/root/work/QE-mls/qe/1d_test",
    )
    plot.hist_2d(
        pred,
        truth,
        ranges=(0, 10000),
        xpad=25,
        save_name="/root/work/QE-mls/qe/2d_test",
    )

    # test grid plots
    # 1D data with -1 to 1 range
    true_list_1d = [np.random.uniform(-1, 1, 10000) for _ in range(6)]
    pred_list_1d = [
        np.random.normal(0, 0.1, 10000) for _ in range(6)
    ]  # Normal distribution centered at 0
    ranges_1d = (-1, 1)  # Set range from -1 to 1
    xlabels_1d = r"$\xi$ [None]"
    titles_1d = [
        r"$\xi^{(0)}_{n}$",
        r"$\xi^{(0)}_{r}$",
        r"$\xi^{(0)}_{k}$",
        r"$\xi^{(1)}_{n}$",
        r"$\xi^{(1)}_{r}$",
        r"$\xi^{(1)}_{k}$",
    ]
    plot.hist_1d_grid(
        true_list_1d,
        pred_list_1d,
        ranges=ranges_1d,
        xlabel=xlabels_1d,
        title=titles_1d,
        bins=30,
        save_name="/root/work/QE-mls/qe/1d_grid_test",
    )

    # 2D data with -1 to 1 range
    true_list_2d = [np.random.uniform(-1, 1, 10000) for _ in range(6)]
    pred_list_2d = [np.random.normal(0, 0.1, 10000) for _ in range(6)]
    ranges_2d = (-1, 1)  # Set range from -1 to 1
    xlabels_2d = r"True $\xi$ [None]"
    ylabels_2d = r"Predicted $\xi$ [None]"
    titles_2d = [
        r"$\xi^{(0)}_{n}$",
        r"$\xi^{(0)}_{r}$",
        r"$\xi^{(0)}_{k}$",
        r"$\xi^{(1)}_{n}$",
        r"$\xi^{(1)}_{r}$",
        r"$\xi^{(1)}_{k}$",
    ]
    plot.hist_2d_grid(
        true_list_2d,
        pred_list_2d,
        ranges=ranges_2d,
        xlabel=xlabels_2d,
        ylabel=ylabels_2d,
        title=titles_2d,
        bins=30,
        save_name="/root/work/QE-mls/qe/2d_grid_test",
    )
    plot.hist_combined_grid(
        true_list_1d,
        pred_list_1d,
        true_list_2d,
        pred_list_2d,
        ranges=(-1, 1),
        xlabel="X",
        title="",
        row1_ylabel="Counts",
        row1_legend=["Seperable", "SM"],
        row2_xlabel="True",
        row2_ylabel="Predicted",
        bins=50,
        xpad=10,
        weights=None,
        save_name="/root/work/QE-mls/qe/grid_test",
        dpi=300,
    )

    print("Done!")
