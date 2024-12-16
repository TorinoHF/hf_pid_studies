"""
Script to download unmerged outputs from hyperloop
"""

import os
import argparse
import math
import multiprocessing

def download_from_directory(task_id, input_directory, is_slim):
    """
    Function to be executed in parallel that downloads all the files for a specific run
    """

    print(f"Processing task {task_id}")
    train_id = input_directory.split(sep="/")[-1]
    if is_slim:
        os.system(f"alien_find alien://{input_directory} "
                  f"AnalysisResults.root -r > outputs_{train_id}.txt")
    else:
        os.system(f"alien_find alien://{input_directory} "
                  f"AOD/*/AnalysisResults.root > outputs_{train_id}.txt")

    check_unmerged = False
    with open(f"outputs_{train_id}.txt") as file:
        lines = [line.rstrip() for line in file]

        if len(lines) == 0:
            check_unmerged = True

    if check_unmerged:
        os.system(f"alien_find alien://{input_directory} "
                  f"*/AnalysisResults.root > outputs_{train_id}.txt")
        with open(f"outputs_{train_id}.txt") as file:
            lines = [line.rstrip() for line in file]

    if not os.path.isdir(train_id):
        os.mkdir(train_id)
    for ifile, line in enumerate(lines):
        os.system(f"alien_cp {line} file:{train_id}/AnalysisResults_{ifile:03d}.root")
        if os.system(f"alien_cp {line.replace('AnalysisResults', 'AO2D')} file:{train_id}/AO2D_{ifile:03d}.root") != 0:
            # AO2D not found, let's delete also AnalysisResults.root
            os.system(f"rm {train_id}/AnalysisResults_{ifile:03d}.root")

    print(f"Processing task {task_id} - DONE")


def download_and_merge(input_file, num_workers, suffix, n_merged_files, is_slim):
    """
    Main function to download and merge output files from hyperloop
    """

    output_directories = []
    with open(input_file, 'r') as file:
        for line in file:
            # Split line by comma or any delimiter
            parts = line.strip().split(',')
            for part in parts:
                output_directories.append(part)

    num_workers = min(num_workers, os.cpu_count())  # Get the number of available CPU cores)

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(
            download_from_directory,
            [(i, directory, is_slim) for i, directory in enumerate(output_directories)]
        )

    files_to_merge = []
    for train_dir in os.listdir("."):
        if os.path.isdir(train_dir) and "hy_" in train_dir:
            for file in os.listdir(train_dir):
                if "AO2D" in file:
                    file_path = os.path.join(train_dir, file)
                    files_to_merge.append(file_path)

    n_files_perbunch = math.ceil(len(files_to_merge) / n_merged_files)
    nbunch = 0
    for ifile, file in enumerate(files_to_merge):
        if ifile % n_files_perbunch == 0:
            nbunch += 1
        os.system(f"echo {file} >> files_to_merge_{nbunch}.txt")

    if nbunch > 1:
        for bunch in range(nbunch):
            os.system(f"o2-aod-merger --input files_to_merge_{bunch}.txt --output "
                      f"AO2D{suffix}_{bunch}.root --max-size 1000000000 --skip-parent-files-list")
    else:
        os.system(f"o2-aod-merger --input files_to_merge_1.txt --output "
                  f"AO2D{suffix}.root --max-size 1000000000 --skip-parent-files-list")

    os.system(f"hadd -f AnalysisResults{suffix}.root hy_*/AnalysisResults*.root")

    os.system(f"rm -r hy_*")
    os.system(f"rm outputs_hy_*")
    os.system(f"rm files_to_merge_*")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Arguments")
    PARSER.add_argument("--input_file", "-i", metavar="text",
                        help="text input file with directories", required=True)
    PARSER.add_argument("--jobs", "-j", type=int, default=20,
                        help="number of workers", required=False)
    PARSER.add_argument("--suffix", "-s", metavar="text",
                        default="_LHC24_pass1_skimmed",
                        help="output file", required=False)
    PARSER.add_argument("--n_merged_files", "-n", type=int,
                        default=1, help="number of output merged AO2D files",
                        required=False)
    PARSER.add_argument("--slim", action="store_true", default=False,
                        help="option for slim outputs on hyperloop", required=False)
    ARGS = PARSER.parse_args()

    download_and_merge(ARGS.input_file, ARGS.jobs, ARGS.suffix, ARGS.n_merged_files, ARGS.slim)
