#!/usr/bin/env bash

INPUT=./data/wikimedia/20250710/wiki.de.txt
OUTDIR=./result
DIM=300
EPOCH=1
THREAD=20
WS=5
NEG=5
T=0.0001
BUCKET=2000000
LR=0.025
MINCOUNT=5

mkdir -p "$OUTDIR"

for MINN in 2 3 4 5 6; do
  for MAXN in $(seq $MINN 6); do
    OUTFILE=$OUTDIR/re-ft-sisg-wiki-de-${MINN}-${MAXN}
    echo "Training with minn=$MINN, maxn=$MAXN → $OUTFILE"
    ./fasttext skipgram \
      -input "$INPUT" \
      -output "$OUTFILE" \
      -dim $DIM \
      -epoch $EPOCH \
      -thread $THREAD \
      -ws $WS \
      -neg $NEG \
      -t $T \
      -bucket $BUCKET \
      -lr $LR \
      -minCount $MINCOUNT \
      -minn $MINN \
      -maxn $MAXN
  done
done
