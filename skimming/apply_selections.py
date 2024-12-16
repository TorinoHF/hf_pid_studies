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

    print("---------------------------------")
    print(f"Loading data from {itree['treefile']}...")
    extension = os.path.splitext(itree["treefile"])[1]
    if extension == ".root":
        # Load the tree
        with uproot.open(itree['treefile'], encoding="utf8") as f:
            df_data = []
            for folder_name, folder in f.items():
                if itree["treename"] in folder_name:
                    df_data.append(folder.arrays(library="pd"))
            df = pd.concat(df_data)
        print("Loaded data from .root file!")
    else:
        df = pd.read_parquet(itree["treefile"])
        print("Loaded data from .parquet file!")
    
    
    df_basename = os.path.splitext(os.path.basename(itree["treefile"]))[0]
    print(f"Selection: {itree['selections'] if itree.get('selections') else 'None'}")
    print(len(df))
    df = df.query(f"{itree['selections']}") if itree.get('selections') else df
    print(len(df))
    print(f"Checking directory: {os.path.dirname(f"{os.path.dirname(itree['treefile'])}/skimmed/")}")
    if not os.path.exists(os.path.dirname(f"{os.path.dirname(itree['treefile'])}/skimmed/")):
        print("Not existing! Creating ...")
        os.makedirs(os.path.dirname(f"{os.path.dirname(itree['treefile'])}/skimmed/"))
    
    if itree.get('suffix'):
        df.to_parquet(f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}_{itree['suffix']}.parquet")
        print(f"Saved skimmed tree to: {f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}_{itree['suffix']}.parquet"}")
    else:
        df.to_parquet(f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}.parquet")
        print(f"Saved skimmed tree to: {f"{os.path.dirname(itree['treefile'])}/skimmed/{df_basename}.parquet"}")