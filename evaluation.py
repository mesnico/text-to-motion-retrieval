import time
import numpy as np
import torch
import numpy
import tqdm
from collections import OrderedDict
from text_similarity_utils.dcg import nDCG

from utils.common import get_motions_and_associated_descriptions, mine_textual_queries


def encode_data(model, data_loader, log_step=100000000, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """

    # switch to evaluate mode
    model.eval()

    # numpy array to keep all the embeddings
    motion_embs = None
    cap_embs = None
    motion_labels = None
    # paths = []
    # descriptions = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ids_pointer = 0
    pbar = tqdm.tqdm(data_loader)
    pbar.set_description('Encoding inference data')
    for i, batch in enumerate(pbar):
        # bs = len(data[0])
        # ids = list(range(ids_pointer, ids_pointer + bs))
        # ids_pointer += bs

        caption, motion, motion_len, labels = batch['desc'], batch['motion'], batch['motion_len'], batch['labels']
        motion = motion.to(device)

        # prepare the inputs
        # X, Y, s2v, path, labels = batch['input'], batch['output'], batch['desc'], batch['path'], batch['labels']
        # labels = torch.stack(labels, dim=1)
        # # paths.extend(path)
        # # descriptions.extend(s2v)

        # pose, trajectory, start_trajectory = X
        # # pose_gt, trajectory_gt, start_trajectory_gt = Y

        # x = torch.cat((trajectory, pose), dim=-1)
        # # y = torch.cat((trajectory_gt, pose_gt), dim=-1)

        # x = x.to(device)
        # # y = y.to(device)
        # if isinstance(s2v, torch.Tensor):
        #     s2v = s2v.to(device)

        # # Transform before the modelye
        # x = pre.transform(x)

        # compute the embeddings
        with torch.no_grad():
            motion_emb, text_emb = model.compute_embeddings(motion, motion_len, caption)

            # initialize the numpy arrays given the size of the embeddings
            if motion_embs is None:
                motion_embs = motion_emb.cpu()
                motion_labels = torch.stack([labels['primary_label_idx'], labels['secondary_label_idx'], labels['top_level_label_idx']], dim=1) if 'null' not in labels else labels['null'].unsqueeze(1).expand(-1, 3)
                cap_embs = text_emb.cpu()
            else:
                motion_embs = torch.cat([motion_embs, motion_emb.cpu()], dim=0)
                lab = torch.stack([labels['primary_label_idx'], labels['secondary_label_idx'], labels['top_level_label_idx']], dim=1) if 'null' not in labels else labels['null'].unsqueeze(1).expand(-1, 3)
                motion_labels = torch.cat([motion_labels, lab], dim=0)
                cap_embs = torch.cat([cap_embs, text_emb.cpu()], dim=0)

            # preserve the embeddings by copying from gpu and converting to numpy
            # img_embs[ids, :] = img_emb.cpu()
            # cap_embs[ids, :] = cap_emb.cpu()

            # measure accuracy and record loss
            # model.forward_loss(None, None, img_emb, cap_emb, img_length, cap_length)

        if (i + 1) % log_step == 0:
            logging('Test: [{0}/{1}]'
                    .format(
                        i, len(data_loader)))

    return motion_embs, cap_embs, motion_labels


def compute_recall(split, queries, motions, labels, dataset, return_all=False, top_k_to_return=50):
    """
    Text->Motion (Motion Search)
    """

    # find the indexes of the useful queries (eliminate duplicates)
    _, useful_queries_idxs = mine_textual_queries(dataset)
    npts = len(useful_queries_idxs)

    # initialize the NDCG metrics
    relevance_methods = ['spacy'] # ['spice', 'spacy']
    dataset_name = dataset.opt.dataset_name
    ndcg_scorer = nDCG(dataset_name, npts, split, relevance_methods=relevance_methods)
    ndcgs = {m: numpy.zeros(npts) for m in relevance_methods}
    ordered_relevances = {m: numpy.zeros((npts, top_k_to_return)) for m in relevance_methods}

    ranks = numpy.zeros(npts).astype(int)
    pbar = tqdm.tqdm(useful_queries_idxs)
    pbar.set_description('Validation')
    idxs = np.zeros((npts, top_k_to_return)).astype(int)
    out_descriptions = []
    
    # find the unique motion ids (assume equal motions are always consecutive)
    _, agg_desc, unique_motion_ids = get_motions_and_associated_descriptions(dataset)

    unique_motion_ids = np.asarray(unique_motion_ids)
    bool_mask = np.zeros(motions.shape[0]).astype(bool)
    bool_mask[unique_motion_ids[1:]] = True
    # diffs = np.insert(bool_mask.astype(int), 0, 0)
    motion_ids = np.cumsum(bool_mask.astype(int))

    for i, q_index in enumerate(pbar):

        # Get query image-url
        query = queries[q_index].unsqueeze(0)
        # Compute scores
        d = torch.mm(query, motions.T)     # 1 x npts
        d = d.cpu().numpy().flatten()

        ordered = numpy.argsort(d)[::-1]
        cleaned_ordered = ordered[np.isin(ordered, unique_motion_ids)]
        idxs[i] = cleaned_ordered[:top_k_to_return]

        # ordered to motion_id
        ordered_motion_id = motion_ids[ordered]

        # assert useful_query_ids[q_index] == i
        res = numpy.where(ordered_motion_id == motion_ids[q_index])[0]

        # the result is the number of unique motion ids before the choosen shot index, which is the minimum among the retrieved ones
        ranks[i] = len(np.unique(ordered_motion_id[:res.min()]))

        # obtain the indexes in the original arrays of the top elements
        _, idx = np.unique(ordered_motion_id, return_index=True)
        unique_order_motion_id = ordered_motion_id[np.sort(idx)]
        top_k_motion_id = unique_order_motion_id[:top_k_to_return]
        idxs[i] = unique_motion_ids[top_k_motion_id]
        out_descriptions.append([agg_desc[i] for i in top_k_motion_id])

        # compute and store ndcg
        ndcg_output = ndcg_scorer.compute_ndcg(i, unique_order_motion_id)
        # ndcgs['spice'][i], ord_rel_spice = ndcg_output[0]['spice'], ndcg_output[1]['spice']
        ndcgs['spacy'][i], ord_rel_spacy = ndcg_output[0]['spacy'], ndcg_output[1]['spacy']
        # ordered_relevances['spice'][i] = ord_rel_spice[:top_k_to_return]
        ordered_relevances['spacy'][i] = ord_rel_spacy[:top_k_to_return]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    ndcg_mean = {k: n.mean() for k, n in ndcgs.items()}
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1

    metrics = {'r1':r1, 'r5': r5, 'r10':r10, 'medr':medr, 'meanr':meanr}

    metrics.update(ndcg_mean)
    
    if return_all:
        return metrics, idxs, out_descriptions, ranks, ordered_relevances
    else:
        return metrics

