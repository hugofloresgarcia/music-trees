""" analyze.py - experiment analysis script"""
import music_trees as mt

from collections import OrderedDict
import glob
from pathlib import Path
import random
from itertools import combinations, permutations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import wilcoxon

ANALYSES_DIR = mt.ROOT_DIR / 'analyses'

ALL_COLORS = ["ff595e", "ffca3a", "8ac926", "1982c4", "6a4c93"]
random.shuffle(ALL_COLORS)


def significance(df: pd.DataFrame, dv: str, iv: str, cond: str):
    """ 
    returns a DataFrame with p values for each condition between independent variables
    """
    # get all possible values for the IV and conditions
    all_trials = df[iv].unique()
    all_conds = list(natsorted(df[cond].unique()))
    pairs = list(permutations(all_trials, 2))

    pvals = []
    for co in all_conds:
        for pair in pairs:
            subset = df[df[cond] == co]
            s1, s2 = pair

            df1 = subset[subset[iv] == s1].copy()
            df2 = subset[subset[iv] == s2].copy()

            stat, p = wilcoxon(df1['value'].values, df2['value'].values)
            pvals.append({
                'a': s1,
                'b': s2,
                cond: co,
                'p': p,
                'stat': stat,
                'significant?': p < 0.01
            })

    return pd.DataFrame(pvals)


def bar_with_error(df: pd.DataFrame, dv: str, iv: str, cond: str, title: str = None) -> plt.figure:
    """
    dv --> dependent variable
    iv --> independent variable
    cond --> conditions (groupings)
    """
    plt.rcParams["figure.figsize"] = (7, 4)

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
    plt.ylabel('value')
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

            dv = 'value'
            iv = 'name'
            cond = 'n_shot'

            errorbar = bar_with_error(subset, dv=dv, iv=iv,
                                      cond=cond, title=metric)
            errorbar.savefig(subdir / f'{metric}.png')

            sig_df = significance(subset, dv=dv, iv=iv, cond=cond)
            sig_df.to_csv(subdir / f'significance-{metric}.csv')

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
