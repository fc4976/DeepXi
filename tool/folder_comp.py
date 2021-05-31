import sys
import os
import glob

if len(sys.argv) != 3:
    print('Usage: test.py ref_folder deg_folder')
    sys.exit(1)

ref_dir = sys.argv[1]
comp_dir = sys.argv[2]

if not os.path.exists(ref_dir):
    print(ref_dir + ' not exist!')
    sys.exit(1)

if not os.path.exists(comp_dir):
    print(comp_dir + ' not exist!')
    sys.exit(1)

comfiles = []
for compfile in glob.iglob(os.path.join(comp_dir, '**/*.wav'), recursive=True):
    filename = os.path.basename(compfile)
    comfiles.append(filename)

reffiles = []
for reffile in glob.iglob(os.path.join(ref_dir, '**/*.wav'), recursive=True):
    filename = os.path.basename(reffile)
    reffiles.append(filename)

samefile = set(comfiles) & set(reffiles)
if (len(samefile) != 0):
    print(samefile)
else:
    print('No same files in %s and %s' %(ref_dir, comp_dir))
