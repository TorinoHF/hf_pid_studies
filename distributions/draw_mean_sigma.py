import re
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def get_label(diff_col_name):
    if diff_col_name == "fPt":
        return '$p_T$ (GeV/$c$)'
    if diff_col_name == "fOccupancyFt0c":
        return 'Occupancy FTOC (arb. units)'
    if diff_col_name == "fCentralityFT0C" or diff_col_name == "fCentralityFT0M":
        return 'Centrality'

def draw_means(dfs_mc, diff_col_name, labels, eff_var):
        
    sort_dfs_data, sort_dfs_mc = [], []
    for df_mc in dfs_mc:
        sort_dfs_mc.append(sort_df_by_interval(df_mc, diff_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    lower_bin_bounds = sort_dfs_mc[0][diff_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_mc[0][diff_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    for df, label in zip(sort_dfs_mc, labels):
        axs[0].errorbar(bin_centers, df[f"{eff_var}_mean"],
                        xerr=bin_widths, label=label, fmt='p')
    axs[0].set_xlabel(get_label(diff_col_name))
    detector = "TOF" if "Tof" in eff_var else "TPC"
    axs[0].set_ylabel(r'Mean N$\sigma^\pi_{' + detector + '}$')
    axs[0].legend()
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    ratios = [np.array(df[f"{eff_var}_mean"]) / np.array(sort_dfs_mc[0][f"{eff_var}_mean"]) for df in sort_dfs_mc[1:]]
    for i_ratio, ratio in enumerate(ratios):
        axs[1].errorbar(bin_centers, ratio, c=f'C{i_ratio+1}',
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(diff_col_name))
    axs[1].set_ylabel(f'Ratio to {labels[0]}') 
    axs[1].axhline(y=1, color='k', linestyle='--')
    return fig

def draw_std(dfs_mc, diff_col_name, labels, eff_var):
        
    sort_dfs_data, sort_dfs_mc = [], []
    for df_mc in dfs_mc:
        sort_dfs_mc.append(sort_df_by_interval(df_mc, diff_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    lower_bin_bounds = sort_dfs_mc[0][diff_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_mc[0][diff_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    for df, label in zip(sort_dfs_mc, labels):
        axs[0].errorbar(bin_centers, df[f"{eff_var}_std"],
                        xerr=bin_widths, label=label, fmt='p')
    axs[0].set_xlabel(get_label(diff_col_name))
    detector = "TOF" if "Tof" in eff_var else "TPC"
    axs[0].set_ylabel(r'RMS N$\sigma^\pi_{' + detector + '}$')
    axs[0].legend()
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    ratios = [np.array(df[f"{eff_var}_std"]) / np.array(sort_dfs_mc[0][f"{eff_var}_std"]) for df in sort_dfs_mc[1:]]
    for i_ratio, ratio in enumerate(ratios):
        axs[1].errorbar(bin_centers, ratio, c=f'C{i_ratio+1}',
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(diff_col_name))
    axs[1].set_ylabel(f'Ratio to {labels[0]}') 
    axs[1].axhline(y=1, color='k', linestyle='--')
    return fig

def draw_ratio_sigma_pos_neg(dfs_pos, dfs_neg, diff_col_name, labels, var_pos, var_neg, labels_df=["positive", "negative"]):
    cmap = plt.get_cmap('tab20')
    sort_dfs_pos, sort_dfs_neg = [], []
    for df_pos in dfs_pos:
        sort_dfs_pos.append(sort_df_by_interval(df_pos, diff_col_name))
    for df_neg in dfs_neg:
        sort_dfs_neg.append(sort_df_by_interval(df_neg, diff_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    lower_bin_bounds = sort_dfs_pos[0][diff_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_pos[0][diff_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    for i_df, (df_pos, df_neg, label) in enumerate(zip(sort_dfs_pos, sort_dfs_neg, labels)):
        axs[0].errorbar(bin_centers, df_pos[f"{var_pos}_std"], c=cmap(i_df*2),
                        xerr=bin_widths, label=f"{label}, {labels_df[0]}", fmt='p')
        axs[0].errorbar(bin_centers, df_neg[f"{var_neg}_std"], c=cmap(i_df*2+1),
                        xerr=bin_widths, label=f"{label}, {labels_df[1]}", fmt='p')
    axs[0].set_xlabel(get_label(diff_col_name))
    detector = "TOF" if "Tof" in var_pos else "TPC"
    axs[0].set_ylabel(r'RMS N$\sigma^\pi_{' + detector + '}$')
    axs[0].legend()
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    ratios = [np.array(df_pos[f"{var_pos}_std"]) / np.array(df_neg[f"{var_neg}_std"]) for df_pos, df_neg in zip(sort_dfs_pos, sort_dfs_neg)]
    for i_ratio, ratio in enumerate(ratios):
        axs[1].errorbar(bin_centers, ratio, c=cmap(i_ratio*2),
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(diff_col_name))
    axs[1].set_ylabel(f'{labels_df[0]}/{labels_df[1]}') 
    axs[1].axhline(y=1, color='k', linestyle='--')
    return fig

def draw_ratio_mean_pos_neg(dfs_pos, dfs_neg, diff_col_name, labels, var_pos, var_neg, labels_df=["positive", "negative"]):
    cmap = plt.get_cmap('tab20')
    sort_dfs_pos, sort_dfs_neg = [], []
    for df_pos in dfs_pos:
        sort_dfs_pos.append(sort_df_by_interval(df_pos, diff_col_name))
    for df_neg in dfs_neg:
        sort_dfs_neg.append(sort_df_by_interval(df_neg, diff_col_name))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    lower_bin_bounds = sort_dfs_pos[0][diff_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_pos[0][diff_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks, rotation=45, ha='right')
    for i_df, (df_pos, df_neg, label) in enumerate(zip(sort_dfs_pos, sort_dfs_neg, labels)):
        axs[0].errorbar(bin_centers, df_pos[f"{var_pos}_mean"], c=cmap(i_df*2),
                        xerr=bin_widths, label=f"{label}, {labels_df[0]}", fmt='p')
        axs[0].errorbar(bin_centers, df_neg[f"{var_neg}_mean"], c=cmap(i_df*2+1),
                        xerr=bin_widths, label=f"{label}, {labels_df[1]}", fmt='p')
    axs[0].set_xlabel(get_label(diff_col_name))
    detector = "TOF" if "Tof" in var_pos else "TPC"
    axs[0].set_ylabel(r'Mean N$\sigma^\pi_{' + detector + '}$')
    axs[0].legend()
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks, rotation=45, ha='right')
    diffs = [np.array(df_pos[f"{var_pos}_mean"]) - np.array(df_neg[f"{var_neg}_mean"]) for df_pos, df_neg in zip(sort_dfs_pos, sort_dfs_neg)]
    for i_ratio, ratio in enumerate(diffs):
        axs[1].errorbar(bin_centers, ratio, c=cmap(i_ratio*2),
                        xerr=bin_widths, label=label, fmt='p')
    axs[1].set_xlabel(get_label(diff_col_name))
    axs[1].set_ylabel(f'{labels_df[0]} - {labels_df[1]}') 
    axs[1].axhline(y=0, color='k', linestyle='--')
    return fig

def draw_plots(input_folder):
    df_neg_pi_mc = pd.read_parquet(f"{input_folder}/neg_pi_eff_df_mc.parquet", engine="pyarrow") 
    df_pos_pi_mc = pd.read_parquet(f"{input_folder}/pos_pi_eff_df_mc.parquet", engine="pyarrow") 
    df_neg_pi = pd.read_parquet(f"{input_folder}/neg_pi_eff_df.parquet", engine="pyarrow") 
    df_pos_pi = pd.read_parquet(f"{input_folder}/pos_pi_eff_df.parquet", engine="pyarrow")

    interval_cols = ['fOccupancyFt0c', 'fPt']

    for col in interval_cols:
        df_neg_pi_mc[col] = df_neg_pi_mc[col].apply(convert_to_interval) 
        df_pos_pi_mc[col] = df_pos_pi_mc[col].apply(convert_to_interval) 
        df_neg_pi[col] = df_neg_pi[col].apply(convert_to_interval) 
        df_pos_pi[col] = df_pos_pi[col].apply(convert_to_interval)

    occ_bins = [0, 4000, 6000, 8000, 100000]
    labels = [f'{occ_min} < Occupancy < {occ_max}' for occ_min, occ_max in zip(occ_bins[:-1], occ_bins[1:])]

    occ_dfs_pos_pi_mc = [df_pos_pi_mc.query(f'fOccupancyFt0c == @pd.Interval({occ_min}, {occ_max}, closed="left")').reset_index(drop=True) for occ_min, occ_max in zip(occ_bins[:-1], occ_bins[1:])]
    occ_dfs_pos_pi = [df_pos_pi.query(f'fOccupancyFt0c == @pd.Interval({occ_min}, {occ_max}, closed="left")').reset_index(drop=True) for occ_min, occ_max in zip(occ_bins[:-1], occ_bins[1:])]
    occ_dfs_neg_pi_mc = [df_neg_pi_mc.query(f'fOccupancyFt0c == @pd.Interval({occ_min}, {occ_max}, closed="left")').reset_index(drop=True) for occ_min, occ_max in zip(occ_bins[:-1], occ_bins[1:])]
    occ_dfs_neg_pi = [df_neg_pi.query(f'fOccupancyFt0c == @pd.Interval({occ_min}, {occ_max}, closed="left")').reset_index(drop=True) for occ_min, occ_max in zip(occ_bins[:-1], occ_bins[1:])]
    
    if not os.path.exists(f"{input_folder}/figures/negative_pion"):
        os.makedirs(f"{input_folder}/figures/negative_pion/mc")
        os.makedirs(f"{input_folder}/figures/negative_pion/data")
    
    if not os.path.exists(f"{input_folder}/figures/positive_pion"):
        os.makedirs(f"{input_folder}/figures/positive_pion/mc")
        os.makedirs(f"{input_folder}/figures/positive_pion/data")
    
    if not os.path.exists(f"{input_folder}/figures/positive_negative"):
        os.makedirs(f"{input_folder}/figures/positive_negative/mc")
        os.makedirs(f"{input_folder}/figures/positive_negative/data")
    
    if not os.path.exists(f"{input_folder}/figures/data_mc"):
        os.makedirs(f"{input_folder}/figures/data_mc/positive")
        os.makedirs(f"{input_folder}/figures/data_mc/negative")

    # Negative pion
    fig = draw_means(occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/mc/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/mc/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/mc/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/mc/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_means(occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/data/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/data/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/data/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/negative_pion/data/tof_rms_pi_pt.png", bbox_inches='tight')

    # Positive pion
    fig = draw_means(occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/mc/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/mc/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/mc/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/mc/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_means(occ_dfs_pos_pi, 'fPt', labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/data/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_means(occ_dfs_pos_pi, 'fPt', labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/data/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_pos_pi, 'fPt', labels, 'fNSigmaTpcPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/data/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_std(occ_dfs_pos_pi, 'fPt', labels, 'fNSigmaTofPosPi')
    fig.savefig(f"{input_folder}/figures/positive_pion/data/tof_rms_pi_pt.png", bbox_inches='tight')

    # Positive/Negative pion
    fig = draw_ratio_mean_pos_neg(occ_dfs_pos_pi_mc, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/mc/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(occ_dfs_pos_pi_mc, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/mc/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(occ_dfs_pos_pi, occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/data/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(occ_dfs_pos_pi, occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/data/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_pos_pi_mc, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/mc/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_pos_pi_mc, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/mc/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_pos_pi, occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTofPosPi', 'fNSigmaTofNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/data/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_pos_pi, occ_dfs_neg_pi, 'fPt', labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcNegPi')
    fig.savefig(f"{input_folder}/figures/positive_negative/data/tpc_rms_pi_pt.png", bbox_inches='tight')

    # Data/MC
    fig = draw_ratio_mean_pos_neg(occ_dfs_pos_pi, occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTofPosPi', 'fNSigmaTofPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/positive/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(occ_dfs_pos_pi, occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/positive/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_pos_pi, occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTofPosPi', 'fNSigmaTofPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/positive/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_pos_pi, occ_dfs_pos_pi_mc, 'fPt', labels, 'fNSigmaTpcPosPi', 'fNSigmaTpcPosPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/positive/tpc_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(occ_dfs_neg_pi, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTofNegPi', 'fNSigmaTofNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/negative/tof_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_mean_pos_neg(occ_dfs_neg_pi, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTpcNegPi', 'fNSigmaTpcNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/negative/tpc_mean_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_neg_pi, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTofNegPi', 'fNSigmaTofNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/negative/tof_rms_pi_pt.png", bbox_inches='tight')
    fig = draw_ratio_sigma_pos_neg(occ_dfs_neg_pi, occ_dfs_neg_pi_mc, 'fPt', labels, 'fNSigmaTpcNegPi', 'fNSigmaTpcNegPi', labels_df=["Data", "MC"])
    fig.savefig(f"{input_folder}/figures/data_mc/negative/tpc_rms_pi_pt.png", bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_folder', help='Input folder')
    args = parser.parse_args()

    draw_plots(args.input_folder)