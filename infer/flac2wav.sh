#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "$0 src_folder [dest_folder]"
    exit 1
fi

src=$1

dest=$2
if [[ $# -eq 1 ]]; then
    echo "use src as dest folder"
    dest=$1
fi

if [[ ! -d "$dest" ]]; then
    mkdir -p "$dest"
fi

old=`pwd`
cd ${dest}
dest=`pwd`

cd ${old}

cd ${src}

for x in *.flac; do
    fn=`echo ${x} | sed s/.flac/.wav/g`
    echo "Converting ${src}/${x} to ${dest}/${fn}"
    sox ${x} -r 16000 ${dest}/${fn}
done
