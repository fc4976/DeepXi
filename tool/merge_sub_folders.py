import sys
import os
import glob
import shutil

if len(sys.argv) != 3:
    print('Usage: merge_sub_folders.py src_folder dest_folder')
    sys.exit(1)

src_folder = sys.argv[1]
dest_folder = sys.argv[2]

if not os.path.exists(src_folder):
    print(src_folder + ' not exist!')
    sys.exit(1)

if not os.path.exists(dest_folder):
    try:
        os.makedirs(dest_folder)
    except OSError:
        raise

for wavfile in glob.iglob(os.path.join(src_folder, '**/*.wav'), recursive=True):
    filename = os.path.basename(wavfile)
    shutil.copy(wavfile, os.path.join(dest_folder, filename))
