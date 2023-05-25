import os
from pathlib import Path
import textwrap
import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import torch
import logging

import argparse

from models.model import MatchingModel

import evaluation
from utils.visualization import plot_3d_motion

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('--set', default='val', type=str,
                    help='Set on which inference is performed.')
    parser.add_argument('--best_on_metric', default='all', help='select snapshot that optimizes this metric')
    parser.add_argument('--override_existing_videos', action='store_true', help='Override existing videos')
    parser.add_argument('--render_descriptions', action='store_true', help='Render descriptions on each video')
    parser.add_argument('--query_ids_to_render', type=int, nargs='+', default=[0], help='ids of the sentence to use as a query for rendering the results')
    # parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')

    args = parser.parse_args()
    log.info(f"Rendering ids: {args.query_ids_to_render}")

    run_path = Path(args.run)
    if not run_path.exists():
        log.warning(f'This path ({run_path}) does not exists. Exiting evaluation.')
        exit(1)

    hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
    OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))

    cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    print(OmegaConf.to_yaml(cfg))

    assert cfg.data.test.name == 'humanml', "Visualization is implemented only on Human-ML3D dataset"

    # ignore the evaluation if checkpoint does not exist or the experiment has not completed.
    last_checkpoint = run_path / 'last.pt'
    if not last_checkpoint.is_file():
        log.warning('Checkpoints not existing. Exiting evaluation...')
        exit(1)
    else:
        checkpoint = torch.load(last_checkpoint, map_location='cpu')
        checkpoint_epoch = checkpoint['epoch']
        if checkpoint_epoch < cfg.optim.epochs - 1:
            log.warning("This run has not been completely executed. Exiting evaluation...")
            exit(1)

    batch_size = cfg.optim.batch_size

    if args.set == 'val':
        dataset_cfg = cfg.data.val
    elif args.set == 'test':
        dataset_cfg = cfg.data.test
    elif args.set == 'train':
        dataset_cfg = cfg.data.train
    else:
        raise ValueError('Set {} not known!'.format(args.set))

    # load datasets and create dataloaders
    dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

    # construct the model
    model = MatchingModel(cfg)
    if torch.cuda.is_available():
        model.cuda()
    # model.double()

    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
    log.info(f"[CKPT]: Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'], strict=True)
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    # perform inference
    motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)
    metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(args.set, caption_feats, motion_feats, motion_labels, dataloader.dataset, return_all=True, top_k_to_return=16)
    log.info(metrics)

    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

    for query_id_to_render in args.query_ids_to_render:
        # load descriptions and motion paths
        all_descriptions = [dataloader.dataset[i]['desc'] for i in range(len(dataloader.dataset))]
        paths = [os.path.join('dataset', 'HumanML3D', 'new_joints', dataloader.dataset[i]['path']) + '.npy' for i in range(len(dataloader.dataset))]

        # find the indexes of the useful queries (eliminate duplicates)
        _, q_idx = np.unique(np.asarray(all_descriptions), return_index=True)
        useful_queries_idxs = np.sort(q_idx)

        output_path = os.path.join('outputs', 'renders', args.set, str(query_id_to_render))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # dump query on file and print it on screen
        with open(os.path.join(output_path, 'desc.txt'), 'w') as f:
            f.write('{}\n{}'.format(
                all_descriptions[useful_queries_idxs[query_id_to_render]],
                ranks[query_id_to_render]
            ))
        # print('Query: {}'.format(all_descriptions[args.query_ids_to_render]))

        # prepare to render
        paths = [paths[idx] for idx in idxs[query_id_to_render]]
        wrapped_descriptions = ['{}; {}; (nDCG:spice={:.2f};spacy={:.2f})'.format(
            i, 
            '; '.join(postproc_descriptions[query_id_to_render][i]),
            ordered_relevances['spice'][query_id_to_render, i],
            ordered_relevances['spacy'][query_id_to_render, i]) 
        for i, idx in enumerate(idxs[query_id_to_render])]

        # chunk the descriptions if too long
        wrapped_descriptions = ['\n'.join(textwrap.wrap(d, 50)) for d in wrapped_descriptions]

        output_filenames = ['{}/{}_{}.mp4'.format(output_path, i, os.path.split(path)[1]) for i, path in enumerate(paths)]

        for npy_file, output_filename, description in zip(paths, output_filenames, wrapped_descriptions):
            data = np.load(npy_file)
            description = description if args.render_descriptions else ''
            plot_3d_motion(output_filename, kinematic_chain, data, title=description, fps=20, radius=4, dist=3, figsize=(5, 5))
        # parallelRender(paths, render_descriptions, output_filenames, skel, feats_kind, data, skip_existing=not args.override_existing_videos)


if __name__ == '__main__':
    main()