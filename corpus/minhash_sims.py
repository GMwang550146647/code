import minhash_funcs
import os
import glob

"""
Parsing / Lemmatization for various corpora
"""

reparse_yelp = True
yelp_json_path = '/scr/nlp/data/yelp-2016/raw/yelp_academic_dataset_review.json'

#temp dirs and such
yelp_parsed = '/scr/thashim/yelp_parsed3'
yelp_split_out = '/scr/thashim/yelp_parsed3_split'
yelp_lsh_out = '/jagupard10/scr1/thashim/yelp3_lsh_chunk' # note - code creates LARGE pickle files in this dir.
yelp_lsh_no_lem_out = '/jagupard10/scr1/thashim/yelp3_lsh_no_lem_chunk' # not used for training set construction.
yelp_adjlist_prefix = '/jagupard10/scr1/thashim/yelp3'
yelp_dataset_output_dir = '/jagupard10/scr1/thashim/yelp3_output'

#derived dir
yelp_split = yelp_parsed+'_orig_gpe.txt'
yelp_split_lemmatized = yelp_parsed+'_lemmatized.txt'


if reparse_yelp:
    minhash_funcs.make_lsh_file_yelp(yelp_json_path, yelp_parsed)
    minhash_funcs.train_test_split(yelp_split , yelp_split_out)
    minhash_funcs.train_test_split(yelp_split_lemmatized, yelp_split_out+'_lemmatized')
    lsh_yelp = minhash_funcs.parallel_make_lsh(yelp_split_out+'_lemmatized.train.txt', yelp_lsh_out, 5, batch_size=100 * 1000)
    lsh_yelp = minhash_funcs.parallel_make_lsh(yelp_split_out + '.train.txt', yelp_lsh_no_lem_out, 5, batch_size=100 * 1000)


"""
Construct adjacency lists for each copora
"""
yelp_list_files = glob.glob(yelp_lsh_out+'*obj')
yelp_seed_set , yelp_seed_sent = minhash_funcs.grab_seeds(yelp_split_out+'_lemmatized.train.txt',50000)
minhash_funcs.generate_adjlist(yelp_seed_sent,500*1000, yelp_adjlist_prefix, yelp_list_files,nproc=5)
minhash_funcs.split_adjlist(yelp_split_out+'.train.txt', yelp_adjlist_prefix+'_adjlist.txt', yelp_adjlist_prefix+'_adjlist')
minhash_funcs.make_test_set(yelp_dataset_output_dir, yelp_adjlist_prefix+'_adjlist_gpe.txt', 5000*1000, lower_jac=0.4)

"""
billion word benchmark
for i in $(seq -f "%05g" 1 99); do cat 'news.en-'$i'-of-00100' >> news.en-cat.txt; done
"""


tmp_dir ='/john9/scr1/thashim/'
onebillion_input = '/scr/nlp/data/onebillion-word/training-monolingual.tokenized.shuffled/news.en-cat.txt'
onebillion_parsed = '/scr/nlp/data/onebillion-word/training-monolingual.tokenized.shuffled/news.en-parsed'
onebillion_split_out = '/scr/thashim/onebillion_split2' # outputs text files with this prefix - plaintext parsed verisons of input
onebillion_lsh_out = tmp_dir+'onebil_lsh_short_chunk' # large output files containing LSH index
onebillion_lsh_no_lem_out = tmp_dir+'onebil_lsh_no_lem_chunk' # not used for training set construction.
onebillion_adjlist_prefix = tmp_dir+'onebil'
onebillion_dataset_output_dir = tmp_dir+'onebillion_split'



minhash_funcs.make_lsh_file_giga(onebillion_input, onebillion_parsed, minlen= 4, maxlen = 20)
minhash_funcs.train_test_split(onebillion_parsed+'_orig_gpe.txt', onebillion_split_out)
minhash_funcs.train_test_split(onebillion_parsed+'_lemmatized.txt', onebillion_split_out+'_lemmatized')
one_lem_lsh = minhash_funcs.parallel_make_lsh(onebillion_split_out+'_lemmatized.train.txt', onebillion_lsh_no_lem_out, 5, batch_size=100 * 1000)
one_short_lsh = minhash_funcs.parallel_make_lsh(onebillion_split_out+'.train.txt', onebillion_lsh_out, 5, batch_size=100*1000)

"""
Construct adjlist for billion word benchmark
"""
one_list_files = glob.glob(onebillion_lsh_out+'*obj')
one_seed_set , one_seed_sent = minhash_funcs.grab_seeds(onebillion_split_out+'.train.txt',1000000)
minhash_funcs.generate_adjlist(one_seed_sent, 500*1000,
                               onebillion_adjlist_prefix,
                               one_list_files, nproc=5)
minhash_funcs.split_adjlist(onebillion_split_out+'.train.txt', onebillion_adjlist_prefix+'_adjlist.txt', onebillion_adjlist_prefix+'_adjlist')
minhash_funcs.make_test_set(onebillion_dataset_output_dir, onebillion_adjlist_prefix+'_adjlist_gpe.txt', 5000*1000, lower_jac=0.4)


one_list_files = glob.glob(onebillion_lsh_no_lem_out+'*obj')
one_seed_set , one_seed_sent = minhash_funcs.grab_seeds(onebillion_split_out+'_lemmatized.train.txt',1000000)
minhash_funcs.generate_adjlist(one_seed_sent, 500*1000,
                               onebillion_adjlist_prefix,
                               one_list_files, nproc=5)
minhash_funcs.split_adjlist(onebillion_split_out+'.train.txt', onebillion_adjlist_prefix+'_lem_adjlist.txt', onebillion_adjlist_prefix+'_lem_adjlist')
minhash_funcs.make_test_set(onebillion_dataset_output_dir+'_lem', onebillion_adjlist_prefix+'_lem_adjlist_gpe.txt', 5000*1000, lower_jac=0.4)
