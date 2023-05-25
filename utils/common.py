import pandas as pd

def mine_textual_queries(dataset):
    data = [(dataset[i]['path'], dataset[i]['desc']) for i in range(len(dataset))]
    df = pd.DataFrame(data, columns=['path', 'desc'])

    # drop duplicated queries
    candidate_queries_df = df.drop_duplicates(subset=['desc'])

    # get queries and its ids
    candidate_queries, candidate_queries_idx = candidate_queries_df['desc'].tolist(), candidate_queries_df.index.tolist()

    return candidate_queries, candidate_queries_idx

def get_motions_and_associated_descriptions(dataset):
    data = [(dataset[i]['path'], dataset[i]['desc']) for i in range(len(dataset))]
    df = pd.DataFrame(data, columns=['path', 'desc'])

    # drop duplicated (motion, desc) and 
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)

    # group by motion
    df = df.groupby(['path'], sort=False, as_index=False).agg({'desc':lambda x: list(x), 'index': 'min'})

    motions, agg_desc, motions_ids = df['path'].tolist(), df['desc'].tolist(), df['index'].tolist()

    return motions, agg_desc, motions_ids