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
    out_noisy = os.path.join(out_dir, 'noisy')
    os.makedirs(out_speech)
    os.makedirs(out_noise)
    os.makedirs(out_noisy)
except OSError:
    raise

print(out_speech, out_noise, out_noisy)

speech_wavs = []
for speech_wav in glob.iglob(os.path.join(speech_dir, '**/*.wav'), recursive=True):
    speech_wavs.append(speech_wav)

noise_wavs = []
for noise_wav in glob.iglob(os.path.join(noise_dir, '**/*.wav'), recursive=True):
    noise_wavs.append(noise_wav)

level_lower = -35
level_upper = -15
snr_min = -5
snr_max = 15

n = 0
random.shuffle(speech_wavs)
for speech_wav in speech_wavs:
    data_speech, fs = audioread(speech_wav)
    if fs == 16000 and len(data_speech) / fs > 2:
        n = n + 1
        flag = True
        while flag:
            noise_wav = random.choice(noise_wavs)
            data_noise, noise_fs = audioread(noise_wav)
            if noise_fs == fs:
                flag = False
        snr = random.randint(snr_min, snr_max)
        data_noisy, data_speech_new, data_noise_new = mix_speech_noise(
            data_speech, data_noise, snr, level_lower, level_upper)
        basename = os.path.basename(speech_wav)
        basenames = os.path.splitext(basename)
        noisy_wav = os.path.join(
            out_noisy, basenames[-2] + '_' + str(snr) + 'dB' + basenames[-1])
        audiowrite(noisy_wav, data_noisy)
        shutil.copy(speech_wav, out_speech)
        shutil.copy(noise_wav, out_noise)
        print('num %d duration %.2f %s' % (n, len(data_noisy) / fs, noisy_wav))
        if n >= number:
            break
