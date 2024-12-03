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
outfile = TFile(f"{selections_cfg['histofile']}", "RECREATE")

for itree in selections_cfg['trees']:


    with uproot.open(f'{itree["treefile"]}') as f:
        print(f.keys())
        # print(f'{itree["foldername"]}')

        if itree.get('subsets'):
            print("GETTING SUBSETS OF SAME FILE")
            for isubset in itree['subsets']:

                single_dfs = []
                for iKey, key in enumerate(f.keys()):
                    if f'{isubset["foldername"]}' in key:
                            print(key)
                            dfData = f[key].arrays(library='pd')
                            single_dfs.append(dfData)
                merged_df = pd.concat([df for df in single_dfs], ignore_index=True)

                outfile.mkdir(isubset['datasetname'])
                outfile.cd(isubset['datasetname'])


                h_mass_before = TH1D('hMass_before', 'hMass_before', isubset['histocfg'][0], isubset['histocfg'][1], isubset['histocfg'][2])
                fill_weigths_before = np.ones(len(merged_df))
                h_mass_before.FillN(len(merged_df)-1, np.asarray(merged_df[f"{isubset['masscol']}"], 'd' ), fill_weigths_before)
                h_mass_before.Write()

                sel_merged_df = merged_df
                sel_merged_df = merged_df.query(f"{isubset['selections']}")

                h_mass_after = TH1D('hMass_after', 'hMass_after', isubset['histocfg'][0], isubset['histocfg'][1], isubset['histocfg'][2])
                fill_weigths_after = np.ones(len(sel_merged_df))
                h_mass_after.FillN(len(sel_merged_df)-1, np.asarray(sel_merged_df[f"{isubset['masscol']}"], 'd' ), fill_weigths_after)
                h_mass_after.Write()

                if not os.path.exists(os.path.dirname(isubset['outfile'])):
                    os.makedirs(os.path.dirname(isubset['outfile']))
                sel_merged_df.to_parquet(f"{isubset['outfile']}")
        else:
            single_dfs = []
            for iKey, key in enumerate(f.keys()):
                if f'{itree["foldername"]}' in key:
                    print(key)
                    dfData = f[key].arrays(library='pd')
                    single_dfs.append(dfData)
            merged_df = pd.concat([df for df in single_dfs], ignore_index=True)

            outfile.mkdir(itree['datasetname'])
            outfile.cd(itree['datasetname'])

            h_mass_before = TH1D('hMass_before', 'hMass_before', itree['histocfg'][0], itree['histocfg'][1], itree['histocfg'][2])
            fill_weigths_before = np.ones(len(merged_df))
            h_mass_before.FillN(len(merged_df)-1, np.asarray(merged_df[f"{itree['masscol']}"], 'd' ), fill_weigths_before)
            h_mass_before.Write()

            sel_merged_df = merged_df
            sel_merged_df = merged_df.query(f"{itree['selections']}")

            h_mass_after = TH1D('hMass_after', 'hMass_after', itree['histocfg'][0], itree['histocfg'][1], itree['histocfg'][2])
            fill_weigths_after = np.ones(len(sel_merged_df))
            h_mass_after.FillN(len(sel_merged_df)-1, np.asarray(sel_merged_df[f"{itree['masscol']}"], 'd' ), fill_weigths_after)
            h_mass_after.Write()

            print(os.path.dirname(itree['outfile']))
            if not os.path.exists(os.path.dirname(itree['outfile'])):
                os.makedirs(os.path.dirname(itree['outfile']))
            print(itree['outfile'])
            sel_merged_df.to_parquet(f"{itree['outfile']}")
            print("CIAO")
    
outfile.Close()