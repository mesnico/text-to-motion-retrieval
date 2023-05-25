import os
from pathlib import Path

import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
import yaml
import traceback
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

import argparse
import utils
# from data import *

from models.model import MatchingModel

from shutil import copyfile
from ast import literal_eval
import evaluation
from utils.checkpoint import CheckpointManager

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg):
    try:
        from omegaconf import OmegaConf; print(OmegaConf.to_yaml(cfg))
        log.info(f"Run path: {Path.cwd()}")
        run_dir = HydraConfig.get().runtime.output_dir

        last_checkpoint = Path(run_dir) / 'last.pt'
        if last_checkpoint.is_file():
            checkpoint = torch.load(last_checkpoint, map_location='cpu')
            checkpoint_epoch = checkpoint['epoch']
            if checkpoint_epoch >= cfg.optim.epochs - 1:
                log.info("This run has already been entirely executed. Exiting....")
                return None

        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
        tb_logger = SummaryWriter(run_dir)

        # torch seed
        torch.manual_seed(cfg.optim.seed)

        batch_size = cfg.optim.batch_size

        # Load datasets and create dataloaders
        train_dataloader = hydra.utils.call(cfg.data.train, batch_size=batch_size)
        val_dataloader = hydra.utils.call(cfg.data.val, batch_size=batch_size)

        # Construct the model
        model = MatchingModel(cfg)
        # model.double()

        # Construct the optimizer and scheduler
        optimizer = hydra.utils.instantiate(cfg.optim.optimizer, model.parameters())
        scheduler = hydra.utils.instantiate(cfg.optim.lr_scheduler, optimizer)

        # # optionally resume from a checkpoint
        start_epoch = 0
        best_metrics = {}
        if cfg.resume:
            filename = 'last.pth'
            assert os.path.isfile(filename), 'Cannot find checkpoint for resuming.'

            log.info("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                model.cuda()

            if cfg.optim.resume:
                log.info("=> loading also optim state from '{}'".format(filename))
                start_epoch = checkpoint['epoch']
                best_metrics = checkpoint['best_metrics']
                # best_rsum = checkpoint['best_rsum']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                # Eiters is used to show logs as the continuation of another
                # training
                # model.Eiters = checkpoint['Eiters']
            log.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(cfg.resume, start_epoch))

        model.train()

        # checkpoint manager
        ckpt_dir = Path(run_dir) / Path('best_models')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_manager = CheckpointManager(ckpt_dir, current_best=best_metrics)

        # Train loop
        mean_loss = 0
        best_rsum = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in tqdm.trange(start_epoch, cfg.optim.epochs):
            progress_bar = tqdm.tqdm(train_dataloader)
            progress_bar.set_description('Train')
            for it, batch in enumerate(progress_bar):
                global_iteration = epoch * len(train_dataloader) + it

                # forward the model
                optimizer.zero_grad()

                # prepare the inputs
                caption, motion, motion_len = batch['desc'], batch['motion'], batch['motion_len']
                # motion = motion.to(device)
                # pose, trajectory, start_trajectory = X
                # # pose_gt, trajectory_gt, start_trajectory_gt = Y

                # x = torch.cat((trajectory, pose), dim=-1).to(device)
                # # y = torch.cat((trajectory_gt, pose_gt), dim=-1).to(device)

                # if isinstance(s2v, torch.Tensor):
                #     s2v = s2v.to(device)

                # # Transform before the model
                # x = pre.transform(x)
                # # y = pre.transform(y)
                # x = x[..., :-4]
                # # y = y[..., :-4]

                loss, monitors = model(motion, motion_len, caption, epoch=epoch)
                loss.backward()

                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                mean_loss += loss.item()

                if global_iteration % cfg.optim.log_every == 0:
                    mean_loss /= cfg.optim.log_every
                    progress_bar.set_postfix(dict(loss='{:.2}'.format(mean_loss)))
                    mean_loss = 0

                tb_logger.add_scalar("Training/Epoch", epoch, global_iteration)
                tb_logger.add_scalar("Training/Loss", loss.item(), global_iteration)
                tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], global_iteration)
                if monitors is not None and len(monitors) > 0:
                    tb_logger.add_scalars("Training/Monitor Values", monitors, global_iteration)

                if global_iteration % cfg.optim.val_freq == 0:
                    # validate
                    metrics = validate(val_dataloader, model)
                    for m, v in metrics.items():
                        tb_logger.add_scalar("Validation/{}".format(m), v, global_iteration)
                    # progress_bar.set_postfix(dict(r1='{:.2}'.format(metrics['r1']), r5='{:.2}'.format(metrics['r5']), meanr='{:.2}'.format(metrics['meanr'])))
                    log.info(metrics)

                    # save only if best on some metric (via CheckpointManager)
                    best_metrics = ckpt_manager.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'metrics': metrics
                    }, metrics, epoch)

                    # save model
                    log.info('Saving model...')
                    checkpoint = {
                        'cfg': cfg,
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_metrics': best_metrics}
                    torch.save(checkpoint, Path(run_dir) / 'last.pt')

            scheduler.step()

    except Exception as error:
        log.error(f"Training ended due to Runtime Error: {error}. Exiting....")
        traceback.print_exc()
        exit(1)

    log.info("Training ended. Exiting....")



def validate(val_dataloader, model):
    model.eval()

    motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, val_dataloader)
    metrics = evaluation.compute_recall('val', caption_feats, motion_feats, motion_labels, val_dataloader.dataset)

    model.train()
    return metrics

if __name__ == '__main__':
    main()