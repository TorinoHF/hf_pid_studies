import argparse
import pandas as pd
import numpy as np
import yaml
import uproot
import os
from ROOT import TFile, TH1D

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('cfg')
args = parser.parse_args()


# Load configuration file
with open(args.cfg, "r") as stream:
    selections_cfg = yaml.safe_load(stream)

for itree in selections_cfg['trees']:

    df_basename = os.path.splitext(os.path.basename(itree["treefile"]))[0]
    df = pd.read_parquet(itree["treefile"])
    print("---------------------------------")
    print(f"Selection: {itree['selections'] if itree.get('selections') else 'None'}")
    sel_df = df.query(f"{itree['selections']}") if itree.get('selections') else df
    print(f"Checking directory: {os.path.dirname(f"{os.path.dirname(itree['treefile'])}/skimmed/")}")
    if not os.path.exists(os.path.dirname(f"{os.path.dirname(itree['treefile'])}/skimmed/")):
        print("Not existing! Creating ...")
        os.makedirs(os.path.dirname(f"{os.path.dirname(itree['treefile'])}/skimmed/"))
    
    if itree.get('suffix'):
        sel_df.to_parquet(f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}_{itree['suffix']}.parquet")
        print(f"Saved skimmed tree to: {f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}_{itree['suffix']}.parquet"}")
    else:
        sel_df.to_parquet(f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}.parquet")
        print(f"Saved skimmed tree to: {f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}.parquet"}")