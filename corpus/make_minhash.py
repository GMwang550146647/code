import sys
import minhash_funcs
import os
import glob
import io
import pickle
import numpy as np
""" 
cl run --request-docker-image=thashim/pylsh:1.0 --request-cpus=5 :corpus :yelpdata 'PYTHONPATH=corpus python corpus/make_minhash.py yelpdata /tmp/yelp_parsed ./yelp_split /tmp/yelp_lsh /tmp/yelp_adj ./'
"""


input_file = '../data/yelp/raw/yelp_academic_dataset_review_part.json'
basedir = '../data/yelp/processed'
parsed_output = os.path.join(basedir, 'yelp_parsed')
test_split_output = os.path.join(basedir, 'yelp_split')
lsh_tmp_prefix = os.path.join(basedir, 'lsh')
adjlist_prefix = os.path.join(basedir, 'yelp_adj')
output_dir = os.path.join(basedir, 'yelp_test')


def main():
    """
    input_file = sys.argv[1]
    parsed_output = sys.argv[2]
    test_split_output =  sys.argv[3]# outputs text files with this prefix - plaintext parsed verisons of input
    lsh_tmp_prefix =  sys.argv[4]# large output files containing LSH index
    adjlist_prefix = sys.argv[5]
    output_dir = sys.argv[6]
    """

    parse_yelp = True
    if parse_yelp:
        minhash_funcs.make_lsh_file_yelp(input_file, parsed_output)
    else:
        minhash_funcs.make_lsh_file_giga(input_file, parsed_output, minlen= 4, maxlen = 20)

    minhash_funcs.train_test_split(parsed_output + '_orig_gpe.txt', test_split_output)
    minhash_funcs.train_test_split(parsed_output + '_lemmatized.txt', test_split_output + '_lemmatized')
    one_short_lsh = minhash_funcs.parallel_make_lsh(test_split_output + '_lemmatized.train.txt', os.path.join(lsh_tmp_prefix, 'lem'),
                                                    5, batch_size=1000)#100 * 1000
    one_short_lsh = minhash_funcs.parallel_make_lsh(test_split_output + '.train.txt', os.path.join(lsh_tmp_prefix, 'nolem'), 5,
                                                    batch_size=1000)#100 * 1000

    """
    Construct adjlist
    """
    one_list_files = glob.glob(lsh_tmp_prefix + '/lem' + '*obj')
    one_seed_set , one_seed_sent = minhash_funcs.grab_seeds(test_split_output + '_lemmatized.train.txt', 1000)
    minhash_funcs.generate_adjlist(one_seed_sent, 500 * 1000, adjlist_prefix, one_list_files, nproc=5)
    minhash_funcs.split_adjlist(test_split_output + '.train.txt', adjlist_prefix + '_adjlist.txt', adjlist_prefix + '_adjlist')
    #minhash_funcs.make_test_set(output_dir, adjlist_prefix + '_adjlist_gpe.txt', 5000 * 1000, lower_jac=0.4)
    minhash_funcs.make_test_set(output_dir, adjlist_prefix + '_adjlist_gpe.txt', 10000, lower_jac=0.4, split_vec=np.array([0.7,0.25,0.05]))
    # grab and dump 200 test set example neighbors here..
    nonlem_lsh = glob.glob(lsh_tmp_prefix + '/nolem' + '*obj')
    dump_test_time_cache(test_split_output+'.test.txt', nonlem_lsh, test_split_output+'.testindex.obj')

def dump_test_time_cache(test_filename, lsh_files, out_file, neval = 200):
    with open(test_filename, 'r', encoding='utf-8') as fopen, open(out_file, 'wb') as fwrite:
        lines = fopen.readlines()
        query_list = []
        lines_list = []
        for i in range(neval):
            lproc = lines[i].strip().split(':')[1]
            lines_list.append(lproc)   # previous neval lines in test file
            query_list.append(lproc.split(' '))  # previous neval lines (token list) in test file
        query_dict = minhash_funcs.lsh_query_parallel(lsh_files, query_list, lines_list, nproc= 5, jac_tr=0.1)
        pickle.dump(query_dict, fwrite, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()