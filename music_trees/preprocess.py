import music_trees as mt
from music_trees.utils.data import get_one_hot
from nussl import AudioSignal

import librosa
import numpy as np


class RandomEffects:

    def __init__(self, effect_chain=None):
        self.effect_chain = effect_chain

    def __call__(self, signal: AudioSignal):
        _validate_audio_signal(signal)
        return mt.utils.effects.augment_from_audio_signal(signal, self.effect_chain)


class LogMelSpec:

    def __init__(self, hop_length: int, win_length: int):
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, entry: dict):
        signal = entry['audio']
        _validate_audio_signal(signal)
        # downmix
        try:
            x = signal.to_mono().audio_data
        except:
            raise ValueError
        x = x[0]
        assert x.ndim == 1

        spec = librosa.feature.melspectrogram(y=x, sr=signal.sample_rate,
                                              hop_length=self.hop_length, win_length=self.win_length,
                                              power=1)

        spec = np.expand_dims(spec, 0)

        entry['audio'] = spec
        return entry

    def __repr__(self):
        return f'logmel-win{self.win_length}-hop{self.hop_length}'


def _validate_audio_signal(signal: AudioSignal):
    assert isinstance(signal, AudioSignal), \
        f'entry["audio"] must be AudioSignal'
    assert signal.has_data
