import os
import glob
import sys
import random
import soundfile
import shutil

if len(sys.argv) != 5:
    print('Usage: move_file.py source_dir dest_dir number duration_in_sec(minimal)')
    sys.exit(1)


root_dir = sys.argv[1]
dest_dir = sys.argv[2]
number = int(sys.argv[3])
sec = int(sys.argv[4])

if not os.path.exists(root_dir):
    print(root_dir + ' not exist!')
    sys.exit(1)

if not os.path.exists(dest_dir):
    try:
        os.makedirs(dest_dir)
    except OSError:
        raise

wavfiles = []
for wavfile in glob.iglob(os.path.join(root_dir, '**/*.wav'), recursive=True):
    wavfiles.append(wavfile)

if number > len(wavfiles):
    print('%s has not enough samples!' % root_dir)
    sys.exit(1)

random.shuffle(wavfiles)

n = 0
for wavfile in wavfiles:
    f = soundfile.SoundFile(wavfile)
    if (f.samplerate == 16000 and len(f) / f.samplerate > sec):
        n = n + 1
        print('num %d len %.2f %s' % (n, len(f) / f.samplerate, wavfile))
        basename = os.path.basename(wavfile)
        shutil.move(wavfile, dest_dir, basename)
        if n >= number:
            break
