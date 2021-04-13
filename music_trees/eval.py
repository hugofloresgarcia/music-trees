import music_trees as mt
from music_trees.tree import MusicTree

import glob

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET = 'katunog'
NUM_WORKERS = 0
N_EPISODES = 1000
N_CLASS = 12
N_QUERY = 16
N_SHOT = (1, 2, 4, 8, 16, 32)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(name: str, version: int):
    exp_dir = mt.train.get_exp_dir(name, version)
    assert exp_dir.exists()

    output_dir = exp_dir / 'tests'
    output_dir.mkdir(exist_ok=True)

    ckpt_path = get_ckpt_path(exp_dir)
    ckpt = torch.load(ckpt_path)
    model = load_model_from_ckpt(ckpt_path)
    model = model.to(DEVICE)

    tree = model.model.tree

    # setup transforms
    audio_tfm = mt.preprocess.LogMelSpec(hop_length=mt.HOP_LENGTH,
                                         win_length=mt.WIN_LENGTH)
    epi_tfm = mt.preprocess.EpisodicTransform()

    all_results = []
    for n_shot in N_SHOT:

        # load our evaluation dataset
        dm = mt.data.MetaDataModule(
            name=DATASET, batch_size=1, num_workers=NUM_WORKERS,
            n_episodes=N_EPISODES, n_class=N_CLASS,
            n_shot=n_shot, n_query=N_QUERY, audio_tfm=audio_tfm,
            epi_tfm=epi_tfm
        )
        dm.setup('test')

        outputs = []
        for index, batch in tqdm(enumerate(dm.test_dataloader())):
            batch = batch2cuda(batch)
            output = model.eval_step(batch, index)
            output['tasks'].append(output['proto_task'])
            output = prune_output(output)
            outputs.append(output)

        results = pd.DataFrame(metrics(outputs, tree))
        results['model'] = f'{name}_v{version}'
        results['n_shot'] = n_shot
        results['n_class'] = N_CLASS
        all_results.append(results)

    all_results = pd.concat(all_results)
    all_results.to_csv(output_dir / 'all_results.csv')

    results_path = mt.RESULTS_DIR / f'{name}-v{version}.csv'
    results_path.parent.mkdir(exist_ok=True)

    all_results.to_csv(results_path)
    for key, val in vars(ckpt['hyper_parameters']['hparams']).items():
        all_results[key] = val
    print(all_results)


def prune_output(output: dict):
    del output['backbone']
    del output['embedding']
    del output['records']
    return output


def get_ckpt_path(exp_dir):
    ckpts = glob.glob(str(exp_dir / 'checkpoints' / '*.ckpt'))
    assert len(ckpts) == 1
    return ckpts[0]


def load_model_from_ckpt(ckpt_path):
    return mt.models.task.MetaTask.load_from_checkpoint(ckpt_path)


def batch2cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
        elif isinstance(v, dict):
            batch[k] = batch2cuda(v)
    return batch


def idx2label(labels: torch.Tensor,  classlist: list):
    return [classlist[l] for l in labels]


def metrics(outputs: dict, tree: MusicTree = None):
    from sklearn.metrics import f1_score

    #  gather a concatenated list of all preds and targets
    task_tags = [t['tag'] for t in outputs[0]['tasks']]
    tasks = {t: {'pred': [], 'target': []} for t in task_tags}

    for o in outputs:
        for t in o['tasks']:
            classlist = t['classlist']
            pred = idx2label(t['pred'],  classlist)
            target = idx2label(t['target'], classlist)
            tasks[t['tag']]['pred'].extend(pred)
            tasks[t['tag']]['target'].extend(target)

    metrics = {tag: {} for tag in tasks.keys()}
    for tag, t in tasks.items():
        classlist = list(set(t['target']))
        metrics[tag]['f1_micro'] = f1_score(t['target'], t['pred'],
                                            average='micro', labels=classlist)
        metrics[tag]['f1_macro'] = f1_score(t['target'], t['pred'],
                                            average='macro',  labels=classlist)
        if tree is not None:
            if tag == 'proto':
                metrics[tag]['hlca'] = np.mean(
                    [tree.hlca(p, tgt) for p, tgt in zip(t['pred'], t['target'])])

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # add training script arguments
    parser.add_argument('--name', type=str, required=True,
                        help='name of the experiment')
    parser.add_argument('--version', type=int, required=True,
                        help='version.')

    args = parser.parse_args()
    evaluate(**vars(args))
