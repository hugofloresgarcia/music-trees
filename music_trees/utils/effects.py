import numpy as np
import sox
import librosa

from instrument_recognition import utils
import audio_utils as au

def _check_audio_types(audio):
    assert audio.ndim == 1, "audio must be mono"
    assert isinstance(audio, np.ndarray)

def get_full_effect_chain():
    effect_chain = ['compand','overdrive', 'eq', 'pitch', 'speed', 
                    'phaser', 'flanger', 'reverb', 'chorus', 'speed', 
                    'lowpass']
    return effect_chain

def augment_from_file_to_file(input_path, output_path, effect_chain=None):
    effect_chain = effect_chain if effect_chain is not None else get_full_effect_chain()
    tfm, effect_params = get_random_transformer(effect_chain)
    tfm.build_file(input_path, output_path)
    return effect_params

#TODO: integrate with audio utils
def augment_from_array_to_array(audio, sr, effect_chain=None):
    effect_chain = effect_chain if effect_chain is not None else get_full_effect_chain()
    tfm, effect_params = get_random_transformer(effect_chain)
    
    # for now, assert that audio is mono and convert to sox format
    audio = au.librosa_input_wrap(audio)
    assert audio.ndim == 1
    audio = np.expand_dims(audio, -1)
    tfm_audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
    audio = np.squeeze(audio, axis=-1)
    tfm_audio = utils.audio.zero_pad(tfm_audio, audio.shape[0])
    tfm_audio = tfm_audio[0:audio.shape[0]]
    # print(tfm_audio.shape)
    tfm_audio = au.librosa_output_wrap(audio)
    # print(tfm_audio.shape)
    return tfm_audio, effect_params

def trim_silence(audio, sr, min_silence_duration=0.3):
    """ trim silence from audio array using sox
    """
    _check_audio_types(audio)
    tfm = sox.Transformer()
    tfm.silence(min_silence_duration=min_silence_duration, 
                buffer_around_silence=False)
    audio = tfm.build_array(input_array=np.expand_dims(audio, axis=1) , sample_rate_in=sr)
    # audio = audio.T[0]
    return audio

def get_randn(mu, std, min=None, max=None):
    """ get a random float, sampled from a normal 
    distribution with mu and std, clipped between min and max
    """
    randn = mu + std * np.random.randn(1)
    return float(randn.clip(min, max))

def choose(collection_like):
    """ given a collection of items, draw one
    (sampled from a uniform distribution)
    """
    min_i = 0
    max_i = len(collection_like)
    idx = int(np.floor(np.random.uniform(min_i, max_i)))
    return collection_like[idx]

def flip_coin(n_times=1):
    """ flip a coin
    """
    result = True
    for _ in range(n_times):
        result = result and choose([True, False])
    return result

def add_effect_with_random_params(tfm, effect_name):
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
            shape=choose(['sine', 'triangle']), 
            interp=choose(['linear', 'quadratic']))
        tfm.flanger(**params)

    elif 'phaser' == effect_name:
        params = dict(
                delay=get_randn(mu=3, std=0.1, min=0, max=5), 
                decay=get_randn(mu=0.4, std=0.05, min=0.1, max=0.5), 
                speed=get_randn(mu=0.5, std=0.1, min=0.1, max=2), 
                modulation_shape=choose(['sinusoidal', 'triangular']))
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
        for filter_num in range(choose([1, 2, 3, 4, 5, 6, 7])): # apply up to 7 filters
            # random log-spaced frequencies between 40 and 20k 
            octave = choose([2 ** i for i in range(8)])
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

    elif 'lowpass' == effect_name:
        params = dict(frequency=get_randn(8000, 2000, min=500, max=12e3))
        tfm.lowpass(**params)

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