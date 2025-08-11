
# h=50, bigram

../rewrite/bin/fastText -train ../data/YFCC100M/train-processing -output ../output/p1_table5/dim50_gram2_lr0.25.bin -size 50 -lr 0.25 -ngram 2 -min-count-vocab 100 -min-count-label 100 -bucket 10000000 -iter 5 -thread 20 -hs 1

../rewrite/bin/test -load-model ../output/p1_table5/dim50_gram2_lr0.25.bin -test-file ../data/YFCC100M/test-processing -topk 1 -answer-threshold 0.0


# h=200, bigram

../rewrite/bin/fastText -train ../data/YFCC100M/train-processing -output ../output/p1_table5/dim200_gram2_lr0.25.bin -size 200 -lr 0.25 -ngram 2 -min-count-vocab 100 -min-count-label 100 -bucket 10000000 -iter 5 -thread 20 -hs 1

../rewrite/bin/test -load-model ../output/p1_table5/dim200_gram2_lr0.25.bin -test-file ../data/YFCC100M/test-processing -topk 1 -answer-threshold 0.0

# h=50 
../rewrite/bin/fastText -train ../data/YFCC100M/train-processing -output ../output/p1_table5/dim50_gram1_lr0.25.bin -size 50 -lr 0.25 -ngram 1 -min-count-vocab 100 -min-count-label 100 -bucket 100000000 -iter 5 -thread 20 -hs 1

../rewrite/bin/test -load-model ../output/p1_table5/dim50_gram1_lr0.25.bin -test-file ../data/YFCC100M/test-processing -topk 1 -answer-threshold 0.0


# h=200

../rewrite/bin/fastText -train ../data/YFCC100M/train-processing -output ../output/p1_table5/dim200_gram1_lr0.25.bin -size 200 -lr 0.25 -ngram 1 -min-count-vocab 100 -min-count-label 100 -bucket 100000000 -iter 5 -thread 20 -hs 1

../rewrite/bin/test -load-model ../output/p1_table5/dim200_gram1_lr0.25.bin -test-file ../data/YFCC100M/test-processing -topk 1 -answer-threshold 0.0

