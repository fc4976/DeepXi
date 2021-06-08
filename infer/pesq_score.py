import soundfile
import sys
import os
import glob
from pesq import pesq
from pystoi import stoi

if len(sys.argv) != 3:
    print('Usage: test.py ref_folder deg_folder')
    sys.exit(1)

ref_dir = sys.argv[1]
deg_dir = sys.argv[2]

if not os.path.exists(ref_dir):
    print(ref_dir + ' not exist!')
    sys.exit(1)

if not os.path.exists(deg_dir):
    print(deg_dir + ' not exist!')
    sys.exit(1)

pesqs = []
stois = []

index = 0
for wavfile in glob.iglob(os.path.join(deg_dir, '**/*.wav'), recursive=True):
    basename = os.path.basename(wavfile)
    filenames = basename.split('_')
    refname = basename[:-len(filenames[-1])-1]+'.wav'
    refname = os.path.join(ref_dir, refname)
    ref, fs = soundfile.read(refname)
    deg, fs = soundfile.read(wavfile)
    deg = deg[:len(ref)]
    pesqs.append(pesq(fs, ref, deg, 'nb'))
    stois.append(100 * stoi(ref, deg, fs, extended=False))
    index = index + 1
    print(' %d' %(index))

print(pesqs)
print(stois)

print('pesq %.4f stoi %.4f' %(sum(pesqs)/len(pesqs), sum(stois)/len(stois)))