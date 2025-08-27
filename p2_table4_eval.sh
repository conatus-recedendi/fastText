#!/bin/bash

for minn in {2..6}; do
  for maxn in $(seq $minn 6); do
    model_file="./result/re-ft-sisg-wiki-de-${minn}-${maxn}.bin"
    echo "[INFO] Running eval with minn=${minn}, maxn=${maxn}, model=${model_file}"

    python eval.py \
      -m "${model_file}" \
      -d ./data/tasks/Gur350_DE.csv \
      --sisg \
      --minn "${minn}" \
      --maxn "${maxn}"
  done
done
