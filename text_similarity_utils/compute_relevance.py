import logging
import numpy as np
import os
import yaml
import tqdm
import argparse
import multiprocessing
from data_loaders.get_data import get_dataset
from text_similarity_utils import spacy_similarity, spice
from ast import literal_eval
import pandas as pd
from utils.common import get_motions_and_associated_descriptions, mine_textual_queries

def clear_desc(q):
    """
    Manually clear a textual description from recurrent terminologies
    """
    q = q.lower()
    q = q.replace('a human', '').replace('a person', '').replace('a human is', '').replace('a person is', '').replace("a person's", '')
    q = q.replace('the human', '').replace('the person', '').replace('the human is', '').replace('the person is', '')
    q = q.replace('someone', '').replace('somebody', '').replace('motion', '')
    q = q.replace('.','').lstrip()
    return q

def compute_relevances_wrt_query(query):
    i, query_caption = query

    # only clear input texts if using spice
    clear = True #compute_relevances_wrt_query.method == 'spice'
    query_caption = clear_desc(query_caption) if clear else query_caption

    if any(compute_relevances_wrt_query.npy_file[i, :] < 0):

        # init the scorer
        scorer = spice.Spice() if compute_relevances_wrt_query.method == 'spice' else spacy_similarity.SpacySimilarity()

        # get motions and associated descriptions
        motions, agg_desc, motions_ids = get_motions_and_associated_descriptions(compute_relevances_wrt_query.dataset)
        if clear:
            agg_desc = [[clear_desc(q) for q in k] for k in agg_desc]

        # find the indexes of the exact-matching motions
        indexes_of_exact_result = [any([query_caption in a for a in b]) for b in agg_desc]
        indexes_of_exact_result = np.asarray(indexes_of_exact_result).nonzero()[0]

        # run the scorer and retrieve the relevances
        if compute_relevances_wrt_query.method == 'spice':
            _, scores = scorer.compute_score(agg_desc, [query_caption])
            relevances = [s['All']['f'] for s in scores]
            relevances = np.array(relevances)
        elif compute_relevances_wrt_query.method == 'spacy':
            relevances = scorer.compute_score(agg_desc, query_caption)

        # patch the spice relevances with the exact results
        relevances[indexes_of_exact_result] = 1.0

        # save on npy file
        compute_relevances_wrt_query.npy_file[i, :] = relevances


def parallel_worker_init(npy_file, dataset, method):
    compute_relevances_wrt_query.npy_file = npy_file
    compute_relevances_wrt_query.dataset = dataset
    compute_relevances_wrt_query.method = method


def main(args):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # load the train dataframe, and associate samples to folds
    # torch.cuda.set_enabled_lms(True)
    # if (torch.cuda.get_enabled_lms()):
    #     torch.cuda.set_limit_lms(11000 * 1024 * 1024)
    #     print('[LMS=On limit=' + str(torch.cuda.get_limit_lms()) + ']')

    dataset = get_dataset(args.dataset, num_frames=0, split=args.set, hml_mode='eval')

    # get queries
    candidate_queries, candidate_queries_idx = mine_textual_queries(dataset)

    # get motions and associated descriptions
    motions, agg_desc, motions_ids = get_motions_and_associated_descriptions(dataset)

    relevance_dir = 'outputs/computed_relevances'
    if not os.path.exists(relevance_dir):
        os.makedirs(relevance_dir)
    relevance_filename = os.path.join(relevance_dir, '{}-{}-{}.npy'.format(args.dataset, args.set, args.method))
    if os.path.isfile(relevance_filename):
        answ = input("Relevances for {} already existing in {}. Continue? (y/n)".format(args.method, relevance_filename))
        if answ != 'y':
            quit()

    # filename = os.path.join(cache_dir,'d_{}.npy'.format(query_img_index))
    n_queries = len(candidate_queries)
    n_motions = len(motions)
    if os.path.isfile(relevance_filename):
        # print('Graph distances file existing for image {}, cache {}! Loading...'.format(query_img_index, cache_name))
        print('Loading existing file {} with shape {} x {}'.format(relevance_filename, n_queries, n_motions))
        npy_file = np.memmap(relevance_filename, dtype=np.float32, shape=(n_queries, n_motions), mode='r+')
    else:
        print('Creating new file {} with shape {} x {}'.format(relevance_filename, n_queries, n_motions))
        npy_file = np.memmap(relevance_filename, dtype=np.float32, shape=(n_queries, n_motions), mode='w+')
        npy_file[:, :] = -1

    # pbar = ProgressBar(widgets=[Percentage(), Bar(), AdaptiveETA()], maxval=n).start()
    print('Starting relevance computation...')
    with multiprocessing.Pool(processes=args.ncpus, initializer=parallel_worker_init,
                              initargs=(npy_file, dataset, args.method)) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(compute_relevances_wrt_query, enumerate(candidate_queries)), total=n_queries):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', default='val', type=str,
                    help='val, test or train')
    parser.add_argument('--method', type=str, default="spacy", help="Scoring method")
    parser.add_argument('--ncpus', type=int, default=2, help="How many gpus to use")

    parser.add_argument('--dataset', type=str, default='kit',
                        help='name of the dataset')

    args = parser.parse_args()
    print(args)

    main(args)
