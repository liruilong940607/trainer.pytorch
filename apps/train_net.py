# Usage:
# python train_net.py -cfg ../configs/example.yaml -- learning_rate 1.0

import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from smplx import SMPL

sys.path.insert(0, '../')
from lib.common.trainer import Trainer
from lib.common.config import get_cfg_defaults
from lib.datasets import AIChoreoDataset
from lib.models import FACTModel
from lib.metrics import visualize


parser = argparse.ArgumentParser()
parser.add_argument(
    '-cfg', '--config_file', type=str, help='path of the yaml config file')
parser.add_argument(
    '--eval_only', action="store_true", help='whether only do evaluation.')
argv = sys.argv[1:sys.argv.index('--')]
args = parser.parse_args(argv)

opts = sys.argv[sys.argv.index('--') + 1:]

# default cfg: defined in 'lib.common.config.py'
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config_file)
# Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
cfg.merge_from_list(opts)
cfg.freeze()


def evaluate(net=None, ckpt_path=None, gen_seq_length=120):
    # set FACT model
    device = "cuda"
    if net is None:
        assert os.path.exists(ckpt_path), ckpt_path
        net = FACTModel().to(device)
        net.load_state_dict(torch.load(ckpt_path)["net"])
    else:
        net = net.module
    net.eval()

    # set dataset
    testval_dataset = AIChoreoDataset(
        "/mnt/data/AIST++/", "/mnt/data/AIST/music", split="testval")
    testval_data_loader = torch.utils.data.DataLoader(
        testval_dataset,
        batch_size=1, shuffle=False,
        num_workers=cfg.num_threads, pin_memory=True, drop_last=False)
    
    # set smpl
    smpl = SMPL(
        model_path=cfg.dataset.smpl_dir, gender='MALE', batch_size=1)

    for data in testval_data_loader:
        # During inference, the `motion` is always the first 120 frames.
        # The `audio` is the entire sequence (40+ secs). The `target` is
        # the remaining motion frames starting from 121-st frame.
        motion, audio, target, seq_name = data
        motion = motion.to(device)
        audio = audio.to(device)
        target = target.to(device)
        # The `output` is the generated motion starting from 121-st frame.
        output = net.inference(motion, audio, gen_seq_length=gen_seq_length)
        visualize(motion=output.cpu().numpy(), smpl_model=smpl)
        # metrics = net.calc_metrics(motion, output, target, audio)
    
    # print('\nEvaluation: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def train(device='cuda'):
    # setup net
    net = FACTModel().to(device)

    # setup trainer
    trainer = Trainer(net, cfg, use_tb=True)
    # load ckpt
    if os.path.exists(cfg.ckpt_path):
        trainer.load_ckpt(cfg.ckpt_path)
    else:
        trainer.logger.info(f'ckpt {cfg.ckpt_path} not found.')

    # set dataset
    train_dataset = AIChoreoDataset(
        "/mnt/data/AIST++/", "/mnt/data/AIST/music", split="train")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_threads, pin_memory=True, drop_last=True)
    trainer.logger.info(
        f'train data size: {len(train_dataset)}; '+
        f'loader size: {len(train_data_loader)};')

    start_iter = trainer.iteration
    start_epoch = trainer.epoch
    # start training
    for epoch in range(start_epoch, cfg.num_epoch):
        trainer.net.train()

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_threads, pin_memory=True, drop_last=True)
        loader = iter(train_data_loader)
        niter = len(train_data_loader)

        epoch_start_time = iter_start_time = time.time()
        for iteration in range(start_iter, niter):
            motion, audio, target, seq_name = next(loader)

            iter_data_time = time.time() - iter_start_time
            global_step = epoch * niter + iteration

            motion = motion.to(device)
            audio = audio.to(device)
            target = target.to(device)
            output = trainer.net(motion, audio)
            loss = trainer.net.module.loss(target, output)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            iter_time = time.time() - iter_start_time
            eta = (niter-iteration) * (time.time()-epoch_start_time) / (iteration-start_iter+1)

            # print
            if iteration % cfg.freq_plot == 0 and iteration > 0:
                trainer.logger.info(
                    f'Name: {cfg.name}|Epoch: {epoch:02d}({iteration:05d}/{niter})|' \
                    +f'dataT: {(iter_data_time):.3f}|' \
                    +f'totalT: {(iter_time):.3f}|'
                    +f'ETA: {int(eta // 60):02d}:{int(eta - 60 * (eta // 60)):02d}|' \
                    +f'Err:{loss.item():.5f}|'
                )
                trainer.tb_writer.add_scalar('data/loss', loss.item(), global_step)

            # save
            if iteration % cfg.freq_save == 0 and iteration > 0:
                trainer.update_ckpt(
                    f'ckpt_{epoch}.pth', epoch, iteration)

            # evaluation
            if iteration % cfg.freq_eval == 0 and iteration > 0:
                trainer.net.eval()
                # -- TODO: change this line below --
                # test(trainer.net.module)
                # ----
                trainer.net.train()

            # end
            iter_start_time = time.time()

        trainer.scheduler.step()
        start_iter = 0


if __name__ == '__main__':
    if args.eval_only:
        evaluate(ckpt_path=cfg.ckpt_path, gen_seq_length=300)
    else:
        train()
