import re
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def convert_to_interval(interval_str):
    match = re.match(r"(\[|\()(\d+(\.\d+)?),\s*(\d+(\.\d+)?)(\]|\))", interval_str)
    if match:
        # Extract the left and right bounds, converting them to float
        left = float(match.group(2))  # Group 2 captures the left bound
        right = float(match.group(4))  # Group 4 captures the right bound
        
        # Determine if the interval is closed on the left or right
        closed = 'left' if match.group(1) == '[' else 'right'
        # Return the interval as a pandas Interval
        return pd.Interval(left, right, closed=closed)
    
    # If the string doesn't match the interval format, return None
    return None  

def sort_df_by_interval(df, col):
    # Use .loc[] to safely modify the DataFrame and avoid the SettingWithCopyWarning
    df.loc[:, "lower_bound"] = df[col].apply(lambda x: x.left)  # Extract left bound
    df_sorted = df.sort_values("lower_bound").drop(columns="lower_bound")  # Sort and drop helper column
    return df_sorted

def get_label(xaxis_col_name):
    if xaxis_col_name == "fPt":
        return '$p_T$ (GeV/$c$)'
    if xaxis_col_name == "fOccupancyFt0c":
        return 'Occupancy FTOC (arb. units)'
    if xaxis_col_name == "fCentralityFT0C" or xaxis_col_name == "fCentralityFT0M":
        return 'Centrality'

def draw_efficiencies(dfs_data, dfs_mc, xaxis_col_name, eff_var, labels):

    cmap = plt.get_cmap('tab20')
    sort_dfs_data, sort_dfs_mc = [], []
    for df_data, df_mc in zip(dfs_data, dfs_mc):
        sort_dfs_data.append(sort_df_by_interval(df_data, xaxis_col_name))
        sort_dfs_mc.append(sort_df_by_interval(df_mc, xaxis_col_name))
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))

    lower_bin_bounds = sort_dfs_data[0][xaxis_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_data[0][xaxis_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    for ax in axs.flat:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, ha='right')

    for i_df, (df_data, df_mc, label) in enumerate(zip(sort_dfs_data, sort_dfs_mc, labels)):
        
        # data and MC leg entries side by side in overlap plots
        data_marker = Line2D([0], [0], color=cmap(i_df*2), marker='o', linestyle='None')
        mc_marker = Line2D([0], [0], color=cmap(i_df*2+1), marker='o', linestyle='None')
        if 'legend_entries' not in locals():
            legend_entries = []
        legend_entries.append(((data_marker, mc_marker), f'{label} data (mc)'))
        
        n_sigma_vals = [3,2,1]
        for i_sigma, n_sigma in enumerate(n_sigma_vals):
            n_sigma_effs_data, n_sigma_effs_uncs_data, n_sigma_effs_mc, n_sigma_effs_uncs_mc = [], [], [], []
            for entry in range(len(df_data)):
                n_sigma_effs_data.append(df_data.loc[entry, eff_var][i_sigma])
                n_sigma_effs_uncs_data.append(df_data.loc[entry, f"{eff_var}_unc"][i_sigma])
                n_sigma_effs_mc.append(df_mc.loc[entry, eff_var][i_sigma])
                n_sigma_effs_uncs_mc.append(df_mc.loc[entry, f"{eff_var}_unc"][i_sigma])

            axs[i_sigma, 0].errorbar(bin_centers, n_sigma_effs_data, c=cmap(i_df*2), 
                                     xerr=bin_widths, fmt='p')
            axs[i_sigma, 0].errorbar(bin_centers, n_sigma_effs_mc, c=cmap(i_df*2+1), 
                                     xerr=bin_widths, fmt='p')
            detector = 'TPC' if 'Tpc' in eff_var else 'TOF'
            var_for_title = r'N$\sigma^\pi_{' + detector + '}$'
            axs[i_sigma, 0].set_title(rf"|{var_for_title}| < {n_sigma}")
            axs[i_sigma, 0].set_ylabel('Efficiency')
            # axs[i_sigma, 0].set_ylim(min_eff.min()/2, 1.2)
            axs[i_sigma, 0].set_xlabel(get_label(xaxis_col_name))
            axs[i_sigma, 0].set_yscale('log')
            axs[i_sigma, 0].grid(True)

            custom_legend = []
            for markers, lbl in legend_entries:
                custom_legend.append((markers, lbl))

            # Add legend to the plot with HandlerTuple for side-by-side markers
            axs[i_sigma, 0].legend([entry[0] for entry in custom_legend], 
                             [entry[1] for entry in custom_legend], 
                             handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')
        
            ratio = np.array(n_sigma_effs_data) / np.array(n_sigma_effs_mc)
            ratio_unc = ratio * np.sqrt((np.array(n_sigma_effs_uncs_data) / np.array(n_sigma_effs_data))**2 + (np.array(n_sigma_effs_uncs_mc) / np.array(n_sigma_effs_mc))**2)

            axs[i_sigma, 1].errorbar(bin_centers, ratio, yerr=ratio_unc, xerr=bin_widths, label=label, fmt='o', color=cmap(i_df*2))

            axs[i_sigma, 1].set_title(rf"Ratio |{var_for_title}| < {n_sigma}")
            axs[i_sigma, 1].set_xlabel(get_label(xaxis_col_name))
            axs[i_sigma, 1].set_ylabel('Data / MC')
            # axs[i_sigma, 1].set_ylim(min_ratio*0.95, max_ratio*1.05)
            axs[i_sigma, 1].legend()
            axs[i_sigma, 1].grid(True)
            axs[i_sigma, 1].axhline(y=1, color='k', linestyle='--')

    return fig

def draw_means(dfs, xaxis_col_name, class_col_name, labels, eff_var):
        
    sort_dfs = []
    for df in dfs:
        sort_dfs.append(sort_df_by_interval(df, class_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{class_col_name} classes', fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])

    lower_bin_bounds = sort_dfs[0][xaxis_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs[0][xaxis_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    for df, label in zip(sort_dfs, labels):
        axs[0].errorbar(bin_centers, df[f"{eff_var}_mean"], yerr=df[f"{eff_var}_mean_unc"],
                        xerr=bin_widths, label=label, fmt='p')
    axs[0].set_xlabel(get_label(xaxis_col_name))
    detector = "TOF" if "Tof" in eff_var else "TPC"
    axs[0].set_ylabel(r'Mean N$\sigma^\pi_{' + detector + '}$')
    axs[0].legend()
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    ratios = [np.array(df[f"{eff_var}_mean"]) / np.array(sort_dfs[0][f"{eff_var}_mean"]) for df in sort_dfs[1:]]
    errs = [abs(ratio)*np.sqrt(
        (np.array(df[f"{eff_var}_mean_unc"])/np.array(df[f"{eff_var}_mean"])) **2 +\
            (np.array(sort_dfs[0][f"{eff_var}_mean_unc"])/np.array(sort_dfs[0][f"{eff_var}_mean"])) **2) for ratio, df in zip(ratios, sort_dfs[1:])]
    for i_ratio, (ratio, unc) in enumerate(zip(ratios, errs)):
        axs[1].errorbar(bin_centers, ratio, yerr=unc, c=f'C{i_ratio+1}',
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(xaxis_col_name))
    axs[1].set_ylabel(f'Ratio to {labels[0]}') 
    axs[1].axhline(y=1, color='k', linestyle='--')
    return fig

def draw_std(dfs, xaxis_col_name, class_col_name, labels, eff_var):
        
    sort_dfs_data, sort_dfs = [], []
    for df in dfs:
        sort_dfs.append(sort_df_by_interval(df, xaxis_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{class_col_name} classes', fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])

    lower_bin_bounds = sort_dfs[0][xaxis_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs[0][xaxis_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    for df, label in zip(sort_dfs, labels):
        axs[0].errorbar(bin_centers, df[f"{eff_var}_std"],
                        xerr=bin_widths, label=label, fmt='p')
    axs[0].set_xlabel(get_label(xaxis_col_name))
    detector = "TOF" if "Tof" in eff_var else "TPC"
    axs[0].set_ylabel(r'RMS N$\sigma^\pi_{' + detector + '}$')
    axs[0].legend()
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    ratios = [np.array(df[f"{eff_var}_std"]) / np.array(sort_dfs[0][f"{eff_var}_std"]) for df in sort_dfs[1:]]
    errs = [abs(ratio) * np.sqrt(
        (np.array(df[f"{eff_var}_std_unc"])/np.array(df[f"{eff_var}_std"])) **2 +\
            (np.array(sort_dfs[0][f"{eff_var}_std_unc"])/np.array(sort_dfs[0][f"{eff_var}_std"])) **2) for ratio, df in zip(ratios, sort_dfs[1:])]
    for i_ratio, (ratio, unc) in enumerate(zip(ratios, errs)):
        axs[1].errorbar(bin_centers, ratio, yerr=unc, c=f'C{i_ratio+1}',
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(xaxis_col_name))
    axs[1].set_ylabel(f'Ratio to {labels[0]}') 
    axs[1].axhline(y=1, color='k', linestyle='--')
    return fig

def draw_ratio_sigma_pos_neg(dfs_pos, dfs_neg, xaxis_col_name, class_col_name, labels, var_pos, var_neg, labels_df=["positive", "negative"]):
    cmap = plt.get_cmap('tab20')
    sort_dfs_pos, sort_dfs_neg = [], []
    for df_pos in dfs_pos:
        sort_dfs_pos.append(sort_df_by_interval(df_pos, xaxis_col_name))
    for df_neg in dfs_neg:
        sort_dfs_neg.append(sort_df_by_interval(df_neg, xaxis_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{class_col_name} classes', fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])

    lower_bin_bounds = sort_dfs_pos[0][xaxis_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_pos[0][xaxis_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    if 'legend_entries' not in locals():
        legend_entries = []
    for i_df, (df_pos, df_neg, label) in enumerate(zip(sort_dfs_pos, sort_dfs_neg, labels)):
        # data and MC leg entries side by side in overlap plots
        pos_marker = Line2D([0], [0], color=cmap(i_df * 2), marker='o', linestyle='None')
        neg_marker = Line2D([0], [0], color=cmap(i_df * 2 + 1), marker='o', linestyle='None')
        legend_entries.append(((pos_marker, neg_marker), f'{label}, {labels_df[0][:3]} ({labels_df[1][:3]})'))

        axs[0].errorbar(bin_centers, df_pos[f"{var_pos}_std"], yerr=df_pos[f"{var_pos}_std_unc"], c=cmap(i_df*2),
                        xerr=bin_widths, fmt='p') # label=f"{label}, {labels_df[0]}")
        axs[0].errorbar(bin_centers, df_neg[f"{var_neg}_std"], yerr=df_neg[f"{var_neg}_std_unc"], c=cmap(i_df*2+1),
                        xerr=bin_widths, fmt='p') # label=f"{label}, {labels_df[1]}")
    axs[0].set_xlabel(get_label(xaxis_col_name))
    detector = "TOF" if "Tof" in var_pos else "TPC"
    axs[0].set_ylabel(r'RMS N$\sigma^\pi_{' + detector + '}$')

    custom_legend = []
    for markers, lbl in legend_entries:
        custom_legend.append((markers, lbl))
    # Add legend to the plot with HandlerTuple for side-by-side markers
    axs[0].legend([entry[0] for entry in custom_legend], 
                     [entry[1] for entry in custom_legend], 
                     handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')

    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    ratios = [np.array(df_pos[f"{var_pos}_std"]) / np.array(df_neg[f"{var_neg}_std"]) for df_pos, df_neg in zip(sort_dfs_pos, sort_dfs_neg)]
    errs = [abs(ratio)*np.sqrt(
        (np.array(df_pos[f"{var_pos}_std_unc"])/np.array(df_pos[f"{var_pos}_std"])) **2 +\
            (np.array(df_neg[f"{var_neg}_std_unc"])/np.array(df_neg[f"{var_neg}_std"])) **2) for ratio, df_pos, df_neg in zip(ratios, sort_dfs_pos, sort_dfs_neg)]
    for i_ratio, (ratio, unc) in enumerate(zip(ratios, errs)):
        axs[1].errorbar(bin_centers, ratio, yerr=unc, c=cmap(i_ratio*2),
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(xaxis_col_name))
    axs[1].set_ylabel(f'{labels_df[0]}/{labels_df[1]}') 
    axs[1].axhline(y=1, color='k', linestyle='--')
    return fig

def draw_ratio_mean_pos_neg(dfs_pos, dfs_neg, xaxis_col_name, class_col_name, labels, var_pos, var_neg, labels_df=["positive", "negative"]):
    cmap = plt.get_cmap('tab20')
    sort_dfs_pos, sort_dfs_neg = [], []
    for df_pos in dfs_pos:
        sort_dfs_pos.append(sort_df_by_interval(df_pos, xaxis_col_name))
    for df_neg in dfs_neg:
        sort_dfs_neg.append(sort_df_by_interval(df_neg, xaxis_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{class_col_name} classes', fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    lower_bin_bounds = sort_dfs_pos[0][xaxis_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_pos[0][xaxis_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    if 'legend_entries' not in locals():
        legend_entries = []
    for i_df, (df_pos, df_neg, label) in enumerate(zip(sort_dfs_pos, sort_dfs_neg, labels)):
        # data and MC leg entries side by side in overlap plots
        pos_marker = Line2D([0], [0], color=cmap(i_df * 2), marker='o', linestyle='None')
        neg_marker = Line2D([0], [0], color=cmap(i_df * 2 + 1), marker='o', linestyle='None')
        legend_entries.append(((pos_marker, neg_marker), f'{label}, {labels_df[0][:3]}({labels_df[1][:3]})'))
        
        axs[0].errorbar(bin_centers, df_pos[f"{var_pos}_mean"], yerr=df_pos[f"{var_pos}_mean_unc"], c=cmap(i_df*2),
                        xerr=bin_widths, fmt='p') #, label=f"{label}, {labels_df[0]}")
        axs[0].errorbar(bin_centers, df_neg[f"{var_neg}_mean"], yerr=df_neg[f"{var_neg}_mean_unc"], c=cmap(i_df*2+1),
                        xerr=bin_widths, fmt='p') #, label=f"{label}, {labels_df[1]}")
    axs[0].set_xlabel(get_label(xaxis_col_name))
    detector = "TOF" if "Tof" in var_pos else "TPC"
    axs[0].set_ylabel(r'Mean N$\sigma^\pi_{' + detector + '}$')
    
    custom_legend = []
    for markers, lbl in legend_entries:
        custom_legend.append((markers, lbl))
    # Add legend to the plot with HandlerTuple for side-by-side markers
    axs[0].legend([entry[0] for entry in custom_legend], 
                     [entry[1] for entry in custom_legend], 
                     handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')
    
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    diffs = [np.array(df_pos[f"{var_pos}_mean"]) - np.array(df_neg[f"{var_neg}_mean"]) for df_pos, df_neg in zip(sort_dfs_pos, sort_dfs_neg)]
    errs = [np.sqrt(df_pos[f"{var_pos}_mean_unc"]**2 + df_neg[f"{var_neg}_mean_unc"]**2) for df_pos, df_neg in zip(sort_dfs_pos, sort_dfs_neg)]
    for i_ratio, (ratio, unc) in enumerate(zip(diffs, errs)):
        axs[1].errorbar(bin_centers, ratio, yerr=unc, c=cmap(i_ratio*2),
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(xaxis_col_name))
    axs[1].set_ylabel(f'{labels_df[0]} - {labels_df[1]}') 
    axs[1].axhline(y=0, color='k', linestyle='--')
    return fig

def draw_plots(outdir, classes_var, xaxis_var, dfs_pos_pi_mc, dfs_pos_pi, dfs_neg_pi_mc, dfs_neg_pi):

    if not os.path.exists(f"{outdir}/negative_pion"):
        os.makedirs(f"{outdir}/negative_pion/mc")
        os.makedirs(f"{outdir}/negative_pion/data")
    
    if not os.path.exists(f"{outdir}/positive_pion"):
        os.makedirs(f"{outdir}/positive_pion/mc")
        os.makedirs(f"{outdir}/positive_pion/data")
    
    if not os.path.exists(f"{outdir}/positive_negative"):
        os.makedirs(f"{outdir}/positive_negative/mc")
        os.makedirs(f"{outdir}/positive_negative/data")
    
    if not os.path.exists(f"{outdir}/data_mc"):
        os.makedirs(f"{outdir}/data_mc/positive")
        os.makedirs(f"{outdir}/data_mc/negative")

    # Negative pion
    print("\n\n ---> Computing NSigma distros for Negative Pi")
    fig = draw_means(dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/negative_pion/mc/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/negative_pion/mc/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/negative_pion/mc/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/negative_pion/mc/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_means(dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/negative_pion/data/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/negative_pion/data/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/negative_pion/data/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/negative_pion/data/tof_rms_pi_pt.png", bbox_inches='tight')

    # Positive pion
    print("\n\n ---> Computing NSigma distros for Positive Pi")
    fig = draw_means(dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{outdir}/positive_pion/mc/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{outdir}/positive_pion/mc/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{outdir}/positive_pion/mc/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{outdir}/positive_pion/mc/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_means(dfs_pos_pi, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{outdir}/positive_pion/data/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(dfs_pos_pi, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{outdir}/positive_pion/data/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_pos_pi, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{outdir}/positive_pion/data/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(dfs_pos_pi, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{outdir}/positive_pion/data/tof_rms_pi_pt.png", bbox_inches='tight')

    # Positive/Negative pion
    print("\n\n ---> Computing NSigma distros comparison for Positive vs Negative Pi")
    fig = draw_ratio_mean_pos_neg(dfs_pos_pi_mc, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/positive_negative/mc/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(dfs_pos_pi_mc, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/positive_negative/mc/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(dfs_pos_pi, dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/positive_negative/data/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(dfs_pos_pi, dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/positive_negative/data/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_pos_pi_mc, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/positive_negative/mc/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_pos_pi_mc, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/positive_negative/mc/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_pos_pi, dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{outdir}/positive_negative/data/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_pos_pi, dfs_neg_pi, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{outdir}/positive_negative/data/tpc_rms_pi_pt.png", bbox_inches='tight')

    # Data/MC
    print("\n\n ---> Computing NSigma distros comparison for Data vs MC")
    fig = draw_ratio_mean_pos_neg(dfs_pos_pi, dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi', 'fNSigmaTofPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/positive/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(dfs_pos_pi, dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/positive/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_pos_pi, dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofPosPi', 'fNSigmaTofPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/positive/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_pos_pi, dfs_pos_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/positive/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(dfs_neg_pi, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofNegPi', 'fNSigmaTofNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/negative/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(dfs_neg_pi, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcNegPi', 'fNSigmaTpcNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/negative/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_neg_pi, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTofNegPi', 'fNSigmaTofNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/negative/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(dfs_neg_pi, dfs_neg_pi_mc, xaxis_var, classes_var, labels, 'fNSigmaTpcNegPi', 'fNSigmaTpcNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{outdir}/data_mc/negative/tpc_rms_pi_pt.png", bbox_inches='tight')

    print(f"\n\n ---> Computing efficiencies for {xaxis_var}")
    if not os.path.exists(f"{outdir}/efficiencies/{xaxis_var}"):
        os.makedirs(f"{outdir}/efficiencies/{xaxis_var}")
    fig = draw_efficiencies(dfs_neg_pi, dfs_neg_pi_mc, xaxis_var, 'fNSigmaTpcNegPi', labels)
    fig.savefig(f"{outdir}/efficiencies/{xaxis_var}/tpc_neg_pi_pt.png", bbox_inches='tight')
    fig = draw_efficiencies(dfs_pos_pi, dfs_pos_pi_mc, xaxis_var, 'fNSigmaTpcPosPi', labels)
    fig.savefig(f"{outdir}/efficiencies/{xaxis_var}/tpc_pos_pi_pt.png", bbox_inches='tight')
    fig = draw_efficiencies(dfs_neg_pi, dfs_neg_pi_mc, xaxis_var, 'fNSigmaTofNegPi', labels)
    fig.savefig(f"{outdir}/efficiencies/{xaxis_var}/tof_neg_pi_pt.png", bbox_inches='tight')
    fig = draw_efficiencies(dfs_pos_pi, dfs_pos_pi_mc, xaxis_var, 'fNSigmaTofPosPi', labels)
    fig.savefig(f"{outdir}/efficiencies/{xaxis_var}/tof_pos_pi_pt.png", bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_folder', help='Input folder')
    parser.add_argument('--classes_var', '-c', default='fCentralityFT0C', help='Variable for classes')
    parser.add_argument('--xaxis_var', '-x', default='fPt', help='Variable for dependency')
    parser.add_argument('--query_var', default='', help='Variable for query')
    parser.add_argument('--query_int', default='', nargs='+', help='Interval to query')
    args = parser.parse_args()

    df_neg_pi_mc = pd.read_parquet(f"{args.input_folder}/neg_pi_eff_df_mc.parquet", engine="pyarrow") 
    df_pos_pi_mc = pd.read_parquet(f"{args.input_folder}/pos_pi_eff_df_mc.parquet", engine="pyarrow") 
    df_neg_pi = pd.read_parquet(f"{args.input_folder}/neg_pi_eff_df.parquet", engine="pyarrow") 
    df_pos_pi = pd.read_parquet(f"{args.input_folder}/pos_pi_eff_df.parquet", engine="pyarrow")

    classes_vars = ['fCentralityFT0C', 'fCentralityFT0M', 'fOccupancyFt0c', 'fOccupancyFT0C']
    classes_var = args.classes_var
    if classes_var not in list(df_neg_pi_mc.columns) and classes_var in classes_vars:
        classes_var = [name for name in classes_vars if name in df_neg_pi_mc.columns][0]

    interval_cols = [classes_var, args.xaxis_var]
    if args.query_var != '':
        interval_cols.append(args.query_var)

    for col in interval_cols:
        print(f"Converting {col} to interval")
        df_neg_pi_mc[col] = df_neg_pi_mc[col].apply(convert_to_interval) 
        df_pos_pi_mc[col] = df_pos_pi_mc[col].apply(convert_to_interval) 
        df_neg_pi[col] = df_neg_pi[col].apply(convert_to_interval) 
        df_pos_pi[col] = df_pos_pi[col].apply(convert_to_interval)

    query_str = ''
    if args.query_var != '':
        query_str = f"_{args.query_var}_{args.query_int[0]}_{args.query_int[1]}"
        query = f'{args.query_var} == @pd.Interval({args.query_int[0]}, {args.query_int[1]}, closed="left")'
        df_neg_pi_mc = df_neg_pi_mc.query(query).reset_index(drop=True) 
        df_pos_pi_mc = df_pos_pi_mc.query(query).reset_index(drop=True) 
        df_neg_pi = df_neg_pi.query(query).reset_index(drop=True) 
        df_pos_pi = df_pos_pi.query(query).reset_index(drop=True)

    bins = sorted(df_pos_pi_mc[classes_var].dropna().unique(), key=lambda x: x.left)
    bin_edges = [interval.left for interval in bins]
    bin_edges.append(bins[-1].right)
    labels = [f'{min}_{max}' for min, max in zip(bin_edges[:-1], bin_edges[1:])]

    dfs_pos_pi_mc = [df_pos_pi_mc.query(f'{classes_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bin_edges[:-1], bin_edges[1:])]
    dfs_pos_pi = [df_pos_pi.query(f'{classes_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bin_edges[:-1], bin_edges[1:])]
    dfs_neg_pi_mc = [df_neg_pi_mc.query(f'{classes_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bin_edges[:-1], bin_edges[1:])]
    dfs_neg_pi = [df_neg_pi.query(f'{classes_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bin_edges[:-1], bin_edges[1:])]

    outdir = f"{args.input_folder}/figures_{classes_var}_xaxis_{args.xaxis_var}{query_str}/"
    draw_plots(outdir, classes_var, args.xaxis_var, dfs_pos_pi_mc, dfs_pos_pi, dfs_neg_pi_mc, dfs_neg_pi)