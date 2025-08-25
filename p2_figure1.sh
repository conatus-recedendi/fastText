#!/usr/bin/env bash

DATADIR=./data/wikimedia/20250710
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
MINN=3
MAXN=6

mkdir -p "$OUTDIR"

for DATA in wiki.de.1%.txt wiki.de.5%.txt wiki.de.20%.txt wiki.de.50%.txt; do
  CASE=$(echo "$DATA" | sed -E 's/^wiki\.de\.([0-9]+%)\.txt/\1/')
  OUTFILE=$OUTDIR/re-ft-sisg-wiki-de-$CASE
  echo "Training on $DATA (minn=$MINN, maxn=$MAXN) → $OUTFILE"
  ./fasttext skipgram \
    -input "$DATADIR/$DATA" \
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
