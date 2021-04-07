import music_trees as mt

import glob
import torch

DATASET = 'katunog'
NUM_WORKERS = 0
N_EPISODES = 10
N_CLASS = 12
N_QUERY = 16
N_SHOT = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model_from_ckpt(exp_dir):
    ckpts = glob.glob(str(exp_dir / 'checkpoints' / '*.ckpt'))
    assert len(ckpts) == 1
    return mt.models.core.ProtoTask.load_from_checkpoint(ckpts[0])


def batch2cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
    return batch


def metrics(output: dict):
    from pytorch_lightning.metrics.functional.f_beta import f1

    result = {}
    for htask in output['tasks']:
        tag = htask['tag']
        num_classes = len(htask['classlist'])
        result[tag] = {
            'f1': f1(htask['pred'], htask['target'], num_classes)
        }

    return result


def evaluate(name: str, version: int,):
    exp_dir = mt.train.get_exp_dir(name, version)
    assert exp_dir.exists()

    output_dir = exp_dir / 'tests'
    output_dir.mkdir(exist_ok=True)

    model = load_model_from_ckpt(exp_dir)
    model = model.to(DEVICE)

    # setup transforms
    audio_tfm = mt.preprocess.LogMelSpec(hop_length=mt.HOP_LENGTH,
                                         win_length=mt.WIN_LENGTH)
    epi_tfm = mt.preprocess.EpisodicTransform()

    # load our evaluation dataset
    dm = mt.data.MetaDataModule(
        name=DATASET, batch_size=1, num_workers=NUM_WORKERS,
        n_episodes=N_EPISODES, n_class=N_CLASS,
        n_shot=N_SHOT, n_query=N_QUERY, audio_tfm=audio_tfm,
        epi_tfm=epi_tfm
    )
    dm.setup('test')

    results = []
    for index, batch in enumerate(dm.test_dataloader()):
        batch = batch2cuda(batch)
        output = model.eval_step(batch, index)
        result = metrics(output)
        results.append(result)
    breakpoint()


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
