#!/users/yulee/micromamba/envs/torch251/bin/python3
import os
import argparse
import sys
import json
import yaml
import torch as th
from torchinfo import summary
import numpy as np
import datetime
import h5py

from nh5 import CombinedDataset, re_namedtuple
from pm import MODE3
from config import get_channel_config
from training_utils import Trainer, ModelWrapper

th.autograd.set_detect_anomaly(True)
th.set_num_threads(16)

if __name__ == '__main__':
    cands = ['e-', 'gamma', 'pi0', 'pi+']
    parser = argparse.ArgumentParser()

    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument("--epoch", type=int, default=200, help='Number of training epochs')
    train_group.add_argument("-em", "--epoch_min", type=int, default=30, help='Minimum epoch threshold for early stopping')
    train_group.add_argument("--balance", type=int, default=None, help='Balance dataset for training')

    model_group = parser.add_argument_group('Model Architecture Arguments')
    model_group.add_argument("--model", type=str, default='Simba', help='Model type (e.g., MLP, Mamba, Conv)')
    model_group.add_argument("--depth", type=int, default=5, help='Depth of the model')
    model_group.add_argument("--encoder_dim", type=int, default=64, help='Encoder and transformer dimension')
    model_group.add_argument("--emb_dim", type=int, default=7, help='Embedding dimension')
    model_group.add_argument("--pma_dim", type=int, default=2, help='PMA dimension')
    model_group.add_argument("--fine_dim", type=int, default=512, help='Fine dimension in the encoder')
    model_group.add_argument("--last_dim", type=int, default=128, help='Last dimension in the encoder')
    model_group.add_argument("--group_size", type=int, default=64, help='Size of group in model')
    model_group.add_argument("--num_group", type=int, default=64, help='Number of groups in model')
    model_group.add_argument("--no_cross", action='store_true', help="Disable cross attention")
    model_group.add_argument("--no_film", action='store_true', help="Disable FiLM layer")

    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument("--num_point", type=int, default=1000, help='Number of points')
    data_group.add_argument("--pool", type=int, default=1, help='Bundling size')
    data_group.add_argument("--input0", type=str, default='Reco', help='Input channel 0')
    data_group.add_argument("--input", type=str, default='DRcalo', help='Input channel')
    data_group.add_argument("--cand", type=str, default=",".join([f'{c}_1-100GeV' for c in cands]), help='Candidate fileset')
    data_group.add_argument("--target", type=str, default="E_gen", help='Regression target variables')
    data_group.add_argument("--target_scale", type=str, default=None, help='Target variable scale')
    data_group.add_argument("--path", type=str, default=None, help='Path to candidate files')

    exec_group = parser.add_argument_group('Execution Arguments')
    exec_group.add_argument("--gpu", type=str, default='2', help='GPU to use')
    exec_group.add_argument("--name", type=str, default='test_id', help='Model name')
    exec_group.add_argument("--eval", action='store_true', help="Set to evaluation mode")
    exec_group.add_argument("--condor", action='store_true', help="Set if running on Condor")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not args.condor:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.condor:
        if not os.path.isdir("save"):
            os.makedirs("save")
        os.environ["TQDM_DISABLE"] = "1"

    print(" ".join(sys.argv[1:]))
    assert th.cuda.is_available()

    start_time = datetime.datetime.now()
    print(start_time)
    name = args.name

    if args.eval:
        data_config, model_config = Trainer.load_checkpoint(name)
        data_cfg = re_namedtuple(yaml.safe_load(data_config))
        cand = data_cfg.input1.cand
    else:
        if "{c}" in args.cand:
            cand = [args.cand.replace("{c}", c) for c in cands]
        else:
            cand = args.cand.split(",")
        target = args.target.split(",")

        CHANNELSET, CHANNELMAX, SCALE = get_channel_config()

        if args.target_scale is None:
            target_scale = [SCALE[t] for t in target]
        else:
            target_scale = [float(t) for t in args.target_scale.split(",")]

        channel0 = CHANNELSET[f'{args.input0}{args.pool}']["CHANNEL"]
        channel1 = CHANNELSET[f'{args.input}{args.pool}']["CHANNEL"]
        channel2 = CHANNELSET['amp']["CHANNEL"]
        in_channel0 = len(channel0)
        in_channel1 = len(channel1)
        channel0max = [CHANNELMAX[c] for c in channel0]
        channel1max = [CHANNELMAX[c] for c in channel1]
        channel2max = [CHANNELMAX[c] for c in channel2]
        num_point0 = 500
        if CHANNELSET[f'{args.input}{args.pool}']["NUM_POINT"] < args.num_point:
            args.num_point = CHANNELSET[f'{args.input}{args.pool}']["NUM_POINT"]
            num_point0 = CHANNELSET[f'{args.input0}{args.pool}']["NUM_POINT"]

        data_config_dict = {
            'input0': {
                'NAME': name,
                'cand': cand,
                'batch_size': 128,
                'channel': channel0,
                'channelmax': channel0max,
                'target': target,
                'target_scale': target_scale,
                'num_point': num_point0,
            },
            'input1': {
                'NAME': name,
                'cand': cand,
                'batch_size': 128,
                'channel': channel1,
                'channelmax': channel1max,
                'target': target,
                'target_scale': target_scale,
                'num_point': args.num_point,
            },
            'input2': {
                'NAME': name,
                'cand': cand,
                'batch_size': 128,
                'channel': channel2,
                'channelmax': channel2max,
                'target': target,
                'target_scale': target_scale,
                'num_point': 1,
            },
        }

        model_config_dict = {
            'model0': {
                'NAME': args.model,
                'in_channel': in_channel0,
                'cls_dim': len(cand),
                'reg_dim': len(target),
                'emb_dim': args.emb_dim,
                'depth': args.depth,
                'group_size': args.group_size,
                'num_group': args.num_group,
                'trans_dim': args.encoder_dim,
                'encoder_dim': args.encoder_dim,
                'pma_dim': args.pma_dim,
                'fine_dim': args.fine_dim,
                'last_dim': args.last_dim,
                'rms_norm': False,
                'drop_path': 0.2,
                'drop_out': 0.1,
            },
            'model': {
                'NAME': args.model,
                'in_channel': in_channel1,
                'cls_dim': len(cand),
                'reg_dim': len(target),
                'emb_dim': args.emb_dim,
                'depth': args.depth,
                'group_size': args.group_size,
                'num_group': args.num_group,
                'trans_dim': args.encoder_dim,
                'encoder_dim': args.encoder_dim,
                'pma_dim': args.pma_dim,
                'fine_dim': args.fine_dim,
                'last_dim': args.last_dim,
                'rms_norm': False,
                'drop_path': 0.2,
                'drop_out': 0.1,
                'cross': not args.no_cross,
                'film': not args.no_film,
            },
        }
        data_config = yaml.dump(data_config_dict)
        model_config = yaml.dump(model_config_dict)

    model_wrapper = ModelWrapper(name, data_config, model_config, args.condor)
    trainer = Trainer(model_wrapper, data_config, model_config, name, args.condor, args.epoch_min)

    data_cfg = re_namedtuple(yaml.safe_load(data_config))
    model_cfgs = re_namedtuple(yaml.safe_load(model_config))
    net = MODE3(model_cfgs)
    if args.condor:
        print("set net")
    model_wrapper.set_net(th.compile(net).cuda())

    if args.eval:
        trainer.load_checkpoint()

    input_size0 = (data_cfg.input1.batch_size, data_cfg.input0.num_point, model_cfgs.model0.in_channel)
    input_size1 = (data_cfg.input1.batch_size, data_cfg.input1.num_point, model_cfgs.model.in_channel)
    input_size2 = (data_cfg.input1.batch_size, 2)
    summary(net, input_size=[input_size0, input_size1, input_size2], depth=6, col_names=("input_size", "output_size", "num_params"))

    balance = args.balance
    if not args.eval:
        print(name)
        train_dataset = CombinedDataset(cand, is_train=True, path=args.path, balance=balance, config=data_cfg)
        val_dataset = CombinedDataset(cand, is_val=True, path=args.path, balance=balance, config=data_cfg)

        train_loader = th.utils.data.DataLoader(train_dataset, batch_size=data_cfg.input1.batch_size, drop_last=True, pin_memory=True, shuffle=True, num_workers=len(cand))
        val_loader = th.utils.data.DataLoader(val_dataset, batch_size=data_cfg.input1.batch_size, num_workers=len(cand))
        print(f"train {len(train_dataset)} validation {len(val_dataset)}")

        trainer.train(train_loader, val_loader, num_epoch=args.epoch)
        del train_dataset, val_dataset
        trainer.load_checkpoint()

    test_dataset = CombinedDataset(cand, is_test=True, path=args.path, balance=balance, config=data_cfg)
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=data_cfg.input1.batch_size, num_workers=len(cand), shuffle=True)
    trainer.test(test_loader)
    print("duration", datetime.datetime.now() - start_time)
