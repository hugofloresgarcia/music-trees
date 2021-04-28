""" analyze.py - experiment analysis script"""
import music_trees as mt

from collections import OrderedDict
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

ANALYSES_DIR = mt.ROOT_DIR / 'analyses'

ALL_COLORS = ["264653", "2a9d8f", "e9c46a", "f4a261", "e76f51"] + \
    ["606c38", "283618", "fefae0", "dda15e", "bc6c25"] + \
    ["0b4eb3", "48ab62", "5a6e8c", "1e375c", "916669"]


def bar_with_error(df: pd.DataFrame, dv: str, iv: str, cond: str, title: str = None) -> plt.figure:
    """
    dv --> dependent variable
    iv --> independent variable
    cond --> conditions (groupings)
    """

    # get all possible values for the IV and conditions
    all_trials = df[iv].unique()
    all_conds = list(natsorted(df[cond].unique()))

    bar_width = 0.25 * 3 / len(all_trials)

    means = OrderedDict((tr, []) for tr in all_trials)
    stds = OrderedDict((tr, []) for tr in all_trials)
    for trial in all_trials:
        for c in all_conds:
            # get the list of all scores per episode
            values = df[(df[iv] == trial) & (df[cond] == c)][dv].values

            means[trial].append(np.mean(values))
            stds[trial].append(np.std(values))

    # make a bar plot for each condition
    bp = np.arange(len(list(means.values())[0]))
    bar_pos = OrderedDict((tr, bp + i*bar_width)
                          for i, tr in enumerate(all_trials))
    for idx, tr in enumerate(all_trials):
        plt.bar(bar_pos[tr], means[tr], yerr=stds[tr], width=bar_width, capsize=12*bar_width,
                color='#'+ALL_COLORS[idx], edgecolor='white', label=tr)

    plt.xlabel(cond)
    plt.xticks(ticks=[i + bar_width for i in range(len(all_conds))],
               labels=all_conds)
    plt.title(title)
    plt.legend()

    fig = plt.gcf()
    plt.close()

    return fig


def analyze(df: pd.DataFrame, name: str):
    """
    run a full analysis and model comparison given
    a DataFrame with multiple results
    """
    output_dir = ANALYSES_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = df['metric'].unique()
    task_tags = df['tag'].unique()

    for tag in task_tags:
        subdir = output_dir / tag
        subdir.mkdir(exist_ok=True)

        for metric in metrics:
            # get the df subset with this
            # metric and tag
            subset = df[(df.tag == tag) & (df.metric == metric)]
            if subset.empty:
                continue

            errorbar = bar_with_error(subset, dv='value', iv='name',
                                      cond='n_shot', title=metric)

            errorbar.savefig(subdir / f'{metric}.png')

    breakpoint()

    raise NotImplementedError


def analyze_folder(path_to_results: str, name: str):
    """
    Will look for .csv files recursively and create a single
    DataFrame for all of them
    """

    filepaths = glob.glob(
        str(Path(path_to_results) / '**/*.csv'), recursive=True)
    df = pd.concat([pd.read_csv(fp) for fp in filepaths], ignore_index=True)

    return analyze(df, name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_results', type=str,
                        help='path to the folder containing result csvs')
    parser.add_argument('name', type=str,
                        help='name of folder with analysis output. will be under /analyses/')
    args = parser.parse_args()

    analyze_folder(**vars(args))
