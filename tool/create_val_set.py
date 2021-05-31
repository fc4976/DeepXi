import audiolib
from audiolib import audioread, audiowrite, snr_mixer, is_clipped, mix_speech_noise
import os
import glob
import sys
import random
import soundfile
import shutil
import numpy as np

if len(sys.argv) != 5:
    print('Usage: add_noise_batch.py speech_folder noise_folder out_folder number')
    sys.exit(1)


speech_dir = sys.argv[1]
noise_dir = sys.argv[2]
out_dir = sys.argv[3]
number = int(sys.argv[4])

if not os.path.exists(speech_dir):
    print(speech_dir + ' not exist!')
    sys.exit(1)
if not os.path.exists(noise_dir):
    print(noise_dir + ' not exist!')
    sys.exit(1)

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
else:
    try:
        os.makedirs(out_dir)
    except OSError:
        raise

try:
    out_speech = os.path.join(out_dir, 'speech')
    out_noise = os.path.join(out_dir, 'noise')
    os.makedirs(out_speech)
    os.makedirs(out_noise)
except OSError:
    raise

speech_wavs = []
for speech_wav in glob.iglob(os.path.join(speech_dir, '**/*.wav'), recursive=True):
    speech_wavs.append(speech_wav)

noise_wavs = []
for noise_wav in glob.iglob(os.path.join(noise_dir, '**/*.wav'), recursive=True):
    noise_wavs.append(noise_wav)

snr_min = -2
snr_max = 25

n = 0
random.shuffle(speech_wavs)
for speech_wav in speech_wavs:
    data_speech, fs = audioread(speech_wav)
    if fs == 16000 and len(data_speech) / fs > 2:
        n = n + 1
        noise_wav = random.choice(noise_wavs)
        data_noise, noise_fs = audioread(noise_wav)
        snr = random.randint(snr_min, snr_max)
        basename = os.path.basename(speech_wav)
        basenames = os.path.splitext(basename)
        noise_wav = os.path.join(
            out_noise, basenames[-2] + '_' + str(snr) + 'dB' + basenames[-1])
        speech_wav = os.path.join(
            out_speech, basenames[-2] + '_' + str(snr) + 'dB' + basenames[-1])
        if len(data_noise) <= len(data_speech):
            used_len = len(data_noise)
            while len(data_speech) > used_len:
                more_len = min(len(data_noise), len(data_speech) - used_len)
                data_noise = np.append(data_noise, data_noise[1:more_len])
                used_len = used_len + more_len
        else:
            index = random.randint(0, len(data_noise) - len(data_speech))
            data_noise = data_noise[index:index+len(data_speech)]
        audiowrite(noise_wav, data_noise)
        audiowrite(speech_wav, data_speech)
        print('num %d noise %.2f speech %.2f %s' % (n, len(data_noise) / fs, len(data_speech) / fs, noise_wav))
        if n >= number:
            break
