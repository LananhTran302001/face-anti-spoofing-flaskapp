import os
import json
import logging
import time
import torch

from fas.losses import (AMSoftmaxLoss, AngleSimpleLinear, SoftTripleLinear, SoftTripleLoss)
from fas.models import mobilenetv2, mobilenetv3_large, mobilenetv3_small


def save_checkpoint(state, filename="my_model.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, net, map_location, optimizer=None, load_optimizer=False, strict=True):
    ''' load a checkpoint of the given model. If model is using for training with imagenet weights provided by
        this project, then delete some wights due to mismatching architectures'''
    print("\n==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        unloaded = net.load_state_dict(checkpoint['state_dict'], strict=strict)
        missing_keys, unexpected_keys = (', '.join(i) for i in unloaded)
    else:
        unloaded = net.load_state_dict(checkpoint, strict=strict)
        missing_keys, unexpected_keys = (', '.join(i) for i in unloaded)
    if missing_keys or unexpected_keys:
        logging.warning(f'THE FOLLOWING KEYS HAVE NOT BEEN LOADED:\n\nmissing keys: {missing_keys}\
            \n\nunexpected keys: {unexpected_keys}\n')
        print('proceed traning ...')
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'epoch' in checkpoint:
        return checkpoint['epoch']

def freeze_layers(model, open_layers):
    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

def build_model(config, device, strict=True, mode='train'):
    ''' build model and change layers depends on loss type'''
    parameters = dict(width_mult=config.model.width_mult,
                    prob_dropout=config.dropout.prob_dropout,
                    type_dropout=config.dropout.type,
                    mu=config.dropout.mu,
                    sigma=config.dropout.sigma,
                    embeding_dim=config.model.embeding_dim,
                    prob_dropout_linear = config.dropout.classifier,
                    theta=config.conv_cd.theta,
                    multi_heads = config.multi_task_learning)

    if config.model.model_type == 'Mobilenet2':
        model = mobilenetv2(**parameters)

        if config.model.pretrained and mode == "train":
            checkpoint_path = config.model.imagenet_weights
            print("Load pretrain weights from ", checkpoint_path, " ...")
            load_checkpoint(checkpoint_path, model, strict=strict, map_location=device)
            if config.model.freeze_layers:
                    for i in config.model.freeze_layers:
                        for param in model.features[i].parameters():
                            param.requires_grad = False
        elif mode == 'convert':
            model.forward = model.forward_to_onnx

        if (config.loss.loss_type == 'amsoftmax') and (config.loss.amsoftmax.margin_type != 'cross_entropy'):
            model.spoofer = AngleSimpleLinear(config.model.embeding_dim, 2)
        elif config.loss.loss_type == 'soft_triple':
            model.spoofer = SoftTripleLinear(config.model.embeding_dim, 2,
                                             num_proxies=config.loss.soft_triple.K)
    else:
        assert config.model.model_type == 'Mobilenet3'
        if config.model.model_size == 'large':
            model = mobilenetv3_large(**parameters)

            if config.model.pretrained and mode == "train":
                checkpoint_path = config.model.imagenet_weights
                print("Load pretrain weights from ", checkpoint_path, " ...")
                load_checkpoint(checkpoint_path, model, strict=strict, map_location=device)
                if config.model.freeze_layers:
                    for i in config.model.freeze_layers:
                        for param in model.features[i].parameters():
                            param.requires_grad = False
            elif mode == 'convert':
                model.forward = model.forward_to_onnx
        else:
            assert config.model.model_size == 'small'
            model = mobilenetv3_small(**parameters)

            if config.model.pretrained and mode == "train":
                checkpoint_path = config.model.imagenet_weights
                print("Load pretrain weights from ", checkpoint_path, " ...")
                load_checkpoint(checkpoint_path, model, strict=strict, map_location=device)
                if config.model.freeze_layers:
                    for i in config.model.freeze_layers:
                        for param in model.features[i].parameters():
                            param.requires_grad = False
            elif mode == 'convert':
                model.forward = model.forward_to_onnx

        if (config.loss.loss_type == 'amsoftmax') and (config.loss.amsoftmax.margin_type != 'cross_entropy'):
            model.scaling = config.loss.amsoftmax.s
            model.spoofer[3] = AngleSimpleLinear(config.model.embeding_dim, 2)
        elif config.loss.loss_type == 'soft_triple':
            model.scaling = config.loss.soft_triple.s
            model.spoofer[3] = SoftTripleLinear(config.model.embeding_dim, 2, num_proxies=config.loss.soft_triple.K)
    return model


def build_criterion(config, device, task='main'):
    if task == 'main':
        if config.loss.loss_type == 'amsoftmax':
            criterion = AMSoftmaxLoss(**config.loss.amsoftmax, device=device)
        elif config.loss.loss_type == 'soft_triple':
            criterion = SoftTripleLoss(**config.loss.soft_triple)
    else:
        assert task == 'rest'
        criterion = AMSoftmaxLoss(margin_type='cross_entropy',
                                  label_smooth=config.loss.amsoftmax.label_smooth,
                                  smoothing=config.loss.amsoftmax.smoothing,
                                  gamma=config.loss.amsoftmax.gamma,
                                  device=device)
    return criterion


class Transform():
    """ class to make diferent transform depends on the label """
    def __init__(self, train_spoof=None, train_real=None, val = None):
        self.train_spoof = train_spoof
        self.train_real = train_real
        self.val_transform = val
        if not all((self.train_spoof, self.train_real)):
            self.train = self.train_spoof or self.train_real
            self.transforms_quantity = 1
        else:
            self.transforms_quantity = 2
    def __call__(self, label, img):
        if self.val_transform:
            return self.val_transform(image=img)
        if self.transforms_quantity == 1:
            return self.train(image=img)
        if label:
            return self.train_spoof(image=img)
        else:
            assert label == 0
            return self.train_real(image=img)


def make_weights(config):
    '''load weights for imbalance dataset to list'''
    if config.dataset != 'celeba-spoof':
        raise NotImplementedError
    with open(os.path.join(config.data.data_root, 'metas/intra_test/items_train.json') , 'r') as f:
        dataset = json.load(f)
    n = len(dataset)
    weights = [0 for i in range(n)]
    keys = list(map(int, list(dataset.keys())))
    keys.sort()
    assert len(keys) == n
    for key in keys:
        label = int(dataset[str(key)]['labels'][43])
        if label:
            weights[int(key)] = 0.1
        else:
            assert label == 0
            weights[int(key)] = 0.2
    assert len(weights) == n
    return n, weights


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()