""" audio effects
"""
import numpy as np
import torch
import random
import sox
import librosa

import music_trees.utils as utils
import audio_utils as au


def _check_audio_types(audio):
    """ check if audio array is sox compatible"""
    assert audio.ndim == 1, "audio must be mono"
    assert isinstance(audio, np.ndarray)


def get_full_effect_chain():
    effect_chain = ['compand', 'overdrive', 'eq', 'pitch', 'speed',
                    'phaser', 'flanger', 'reverb', 'chorus', 'speed']
    return effect_chain


def augment_from_file_to_file(input_path, output_path, effect_chain=None):
    """ augment an audio file, and write to a new audio file. 
    see get_full_event_chain() to view available effects. if none, uses all effects
    """
    effect_chain = effect_chain if effect_chain is not None else get_full_effect_chain()
    tfm, effect_params = get_random_transformer(effect_chain)
    tfm.build_file(input_path, output_path)
    return effect_params


def augment_from_array_to_array(audio, sr, effect_chain=None):
    """ augment an np array, and write to a new np array file. 
    the audio should be shape (channels, samples)
    see get_full_event_chain() to view available effects. if none, uses all effects
    """
    effect_chain = effect_chain if effect_chain is not None else get_full_effect_chain()
    tfm, effect_params = get_random_transformer(effect_chain)

    # for now, assert that audio is mono and convert to sox format
    audio = au.librosa_input_wrap(audio)
    assert audio.ndim == 1

    audio = np.expand_dims(audio, -1)
    tfm_audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
    audio = np.squeeze(audio, axis=-1)

    tfm_audio = au.librosa_output_wrap(audio)
    tfm_audio = utils.audio.zero_pad(tfm_audio, audio.shape[-1])
    tfm_audio = tfm_audio[:, 0:audio.shape[-1]]

    return tfm_audio, effect_params


def augment_from_audio_signal(signal, effect_chain=None):
    """ augment a nussl audiosignal, and return an augmented audiosignal. 
    the audio should be shape (channels, samples)
    see get_full_event_chain() to view available effects. if none, uses all effects
    """
    signal.audio_data, effect_params = augment_from_array_to_array(signal.audio_data,
                                                                   signal.sample_rate, effect_chain)
    signal._effect_params = effect_params
    return signal


def trim_silence(audio, sr, min_silence_duration=0.3):
    """ trim silence from audio array using sox
    """
    _check_audio_types(audio)
    tfm = sox.Transformer()
    tfm.silence(min_silence_duration=min_silence_duration,
                buffer_around_silence=False)
    audio = tfm.build_array(input_array=np.expand_dims(
        audio, axis=1), sample_rate_in=sr)
    return audio


def get_randn(mu, std, min=None, max=None):
    """ get a random float, sampled from a normal 
    distribution with mu and std, clipped between min and max
    """
    randn = mu + std * torch.randn(1).numpy()
    return float(randn.clip(min, max))


def flip_coin(n_times=1):
    """ flip a coin
    """
    result = True
    for _ in range(n_times):
        result = result and random.choice([True, False])
    return result


def add_effect_with_random_params(tfm, effect_name, sample_rate=16000):
    """ add an effect with random params (that make sense somewhat)
    to the transformer tfm
    returns:
        tfm: transformer object
        param_dict: params applied 
    """
    if 'flanger' == effect_name:
        params = dict(delay=get_randn(mu=0, std=5, min=0, max=30),
                      depth=get_randn(mu=2, std=5, min=0, max=10),
                      regen=get_randn(mu=0, std=10, min=-95, max=95),
                      width=get_randn(mu=71, std=5, min=0, max=100),
                      speed=get_randn(mu=0.5, std=0.2, min=0.1, max=10),
                      phase=get_randn(mu=25, std=7, min=0, max=100),
                      shape=random.choice(['sine', 'triangle']),
                      interp=random.choice(['linear', 'quadratic']))
        tfm.flanger(**params)

    elif 'phaser' == effect_name:
        params = dict(
            delay=get_randn(mu=3, std=0.1, min=0, max=5),
            decay=get_randn(mu=0.4, std=0.05, min=0.1, max=0.5),
            speed=get_randn(mu=0.5, std=0.1, min=0.1, max=2),
            modulation_shape=random.choice(['sinusoidal', 'triangular']))
        tfm.phaser(**params)

    elif 'overdrive' == effect_name:
        params = dict(
            gain_db=get_randn(mu=10, std=5, min=0, max=40),
            colour=get_randn(mu=20, std=5, min=0, max=40))
        tfm.overdrive(**params)

    elif 'compand' == effect_name:
        params = dict(amount=get_randn(mu=75, std=20, min=0, max=100))
        tfm.contrast(**params)

    elif 'eq' == effect_name:
        params = {}
        # apply up to 7 filters
        for filter_num in range(random.choice([1, 2, 3, 4, 5, 6, 7])):
            # random log-spaced frequencies between 40 and 20k
            octave = random.choice([2 ** i for i in range(5)])
            sub_params = dict(
                frequency=get_randn(mu=60, std=10, min=40, max=80) * octave,
                gain_db=get_randn(mu=0, std=6, min=-24, max=24),
                width_q=get_randn(mu=0.707, std=0.1415, min=0.3, max=1.4))
            tfm.equalizer(**sub_params)

            params[filter_num] = sub_params

    elif 'pitch' == effect_name:
        params = dict(n_semitones=get_randn(mu=0, std=3, min=-7, max=7))
        tfm.pitch(**params)

    elif 'reverb' == effect_name:
        params = dict(
            reverberance=get_randn(mu=50, std=15, min=0, max=100),
            high_freq_damping=get_randn(mu=50, std=25, min=0, max=100),
            room_scale=get_randn(mu=50, std=25, min=0, max=100),
            wet_gain=get_randn(mu=0, std=2, min=-3, max=3))
        tfm.reverb(**params)

    # elif 'lowpass' == effect_name:
    #     params = dict(frequency=get_randn(
    #         8000, 2000, min=500, max=sample_rate//2))
    #     tfm.lowpass(**params)

    elif 'chorus' == effect_name:
        params = dict(gain_in=get_randn(0.3, 0.1, 0.1, 0.5),
                      gain_out=get_randn(0.8, 0.1, 0.5, 1),
                      n_voices=int(get_randn(3, 1, 2, 8)))
        tfm.chorus(**params)

    elif 'speed' == effect_name:
        params = dict(factor=get_randn(mu=1, std=0.125, min=0.75, max=1.25))
        tfm.speed(**params)

    return tfm, params


def get_random_transformer(effect_chain):
    """ 
    params:
        effect_chain (list of str): list of effects to apply
    """
    tfm = sox.Transformer()

    effect_params = {}
    for e in effect_chain:
        if flip_coin():
            continue
        tfm, params = add_effect_with_random_params(tfm, e)
        effect_params[e] = params

    return tfm, effect_params
