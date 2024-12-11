import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import ROOT

ROOT.gROOT.SetBatch(True)

# Custom function for the lower triangle: heatmap
def lower_triangle_heatmap(x, y, **kwargs):
    data = kwargs.pop('data', None)
    if x.name != y.name:  # No diagonal elements
        # Plot correlation coefficient
        corr = data.corr()
        corr_value = corr.loc[y.name, x.name]
        sns.heatmap(
            pd.DataFrame([[corr_value]]),  # Single value as 1x1 heatmap
            vmin=-1, vmax=1, annot=True, cbar=False, square=True, cmap="coolwarm",
            xticklabels=False, yticklabels=False, ax=plt.gca()
        )

# Custom function for the upper triangle: scatter plot
def upper_triangle_scatter(x, y, **kwargs):
    if x.name != y.name:  # Avoid diagonal elements
        sns.scatterplot(x=x, y=y, **kwargs)

def draw_correlation_pt(data_df, suffix, pt_min, pt_max, cfg, dau):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # Create PairGrid
    g = sns.PairGrid(data_df, vars=[*cfg['dau_pt_var_names'], cfg['mother_pt_var_name']])

    # Map functions to the lower and upper triangle
    g.map_lower(lower_triangle_heatmap, cmap="coolwarm", data=data_df)
    g.map_upper(sns.histplot, bins=40)
    g.map_diag(sns.histplot, bins=40)

    # Adjust layout
    plt.tight_layout()

    output_dir = os.path.join(cfg['output']['dir'], f'{dau}/{pt_min*10:.0f}_{pt_max*10:.0f}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(
        output_dir,
        f'corr_matrix_{suffix}.png'
    )
    g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
