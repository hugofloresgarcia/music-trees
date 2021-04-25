""" hyperparam search! :)
"""
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune
import numpy as np
from functools import partial
import music_trees as mt


SEEDS = [mt.SEED]

# default hyperparameters go here
DEFAULTS = {
    'name': 'auto',

    'model_name': 'hprotonet',
    'height': 0,
    'd_root': 128,
    'loss_alpha': 0.5,

    'dataset': 'mdb-aug',
    'num_workers': 20,
    'learning_rate': 0.03
}

CONFIGS = {
    'data-aug': {
        'dataset': tune.grid_search(['mdb-aug', 'mdb']),
    },
    'tree-height': {
        'height': tune.grid_search(list(range(6))),
    },
    'd_root': {
        'd_root': tune.grid_search([64, 128, 256, 512]),
    },
    'loss-exp': {
        'loss_weight_fn': 'exp',
        'loss_alpha': list(np.arange(-4, 4.25, 0.25)),
    },

}


class Experiment:
    def __init__(self, name: str, defaults: dict, config: dict, gpu_fraction: float):
        self.name = name
        self.config = config
        self.hparams = argparse.Namespace(
            **{k: v for k, v in defaults.items() if k not in config})
        self.gpu_fraction = gpu_fraction


def hparams2args(hparams):
    args = []
    for k, v in vars(hparams).items():
        args.append(f'--{k}')
        args.append(f'{v}')
    return args


def run_trial(config, **kwargs):
    hparams = argparse.Namespace(**kwargs)
    hparams.__dict__.update(config)

    hparams.name = f'{ {k: v for k, v in config.items()}}'

    parser = mt.train.load_parser(known_args=hparams2args(hparams))
    del hparams.parent_name
    del hparams.checkpoint_dir
    hparams = parser.parse_args(args=hparams2args(hparams))

    return mt.train.train(hparams, use_ray=True)


def run_experiment(exp, num_samples):

    scheduler = ASHAScheduler(
        metric="f1/protonet/val",
        mode="max",
        max_t=mt.train.MAX_EPISODES,
        grace_period=mt.train.MAX_EPISODES // 32,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["f1/protonet/train",
                        "f1/protonet/val", ])

    result = tune.run(
        partial(run_trial, **vars(exp.hparams)),
        name=exp.name,
        local_dir=mt.RUNS_DIR,
        resources_per_trial={"cpu": 1, "gpu": exp.gpu_fraction},
        config=exp.config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    df = result.results_df
    df.to_csv(str(mt.train.get_exp_dir(exp.hparams.name,
              exp.hparams.version)/'ray-results.csv'))


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--gpu_capacity', type=float, default=1.0)

    args = parser.parse_args()

    parent_name = args.name + '-' + datetime.now().strftime("%m.%d.%Y")
    mt.RUNS_DIR = mt.RUNS_DIR / parent_name
    mt.TQDM_DISABLE = True

    exp = Experiment(name=args.name, defaults=DEFAULTS,
                     config=CONFIGS[args.name], gpu_fraction=args.gpu_capacity)
    exp.hparams.parent_name = parent_name

    run_experiment(exp, args.num_samples)