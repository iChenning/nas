import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import utils.models as models
import utils.metrics as metrics


def load_normal(load_path):
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def load_vit(load_path, encoder):
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if 'embeddings.position_embeddings' in k:
            posemb = v
            posemb_new = encoder.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                state_dict_new[k] = posemb
            else:
                ntok_new = posemb_new.size(1)

                if encoder.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('=>vit load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                posemb_grid = posemb_grid.unsqueeze(0)
                posemb_grid = posemb_grid.permute(0, 3, 1, 2)
                posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear', align_corners=False)
                posemb_grid = posemb_grid[0]
                posemb_grid = posemb_grid.permute(1, 2, 0)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)

                v = torch.cat((posemb_tok, posemb_grid), dim=1)
                state_dict_new[k] = v
        else:
            state_dict_new[k] = v
    return state_dict_new


def import_encoder(opt):
    """
    Encoder only used to generate feature, it doesn't have fc.
    The feature generated from encoder didn't normalize.
    """
    if 'resnet' in opt.encoder.name:
        pretrained = True if opt.encoder.init_type == 'pytorch' else False
        encoder = models.__dict__[opt.encoder.name](pretrained=pretrained)
        encoder.fc = torch.nn.Sequential()
        if opt.encoder.init_type == 'load':
            new_state_dict = load_normal(opt.encoder.load_path)
            encoder.load_state_dict(new_state_dict)

    elif 'vit' in opt.encoder.name:
        encoder = models.__dict__[opt.encoder.name](data_size=opt.data.data_size)
        if opt.encoder.init_type == 'load':
            new_state_dict = load_vit(opt.encoder.load_path, encoder)
            encoder.load_state_dict(new_state_dict)

    else:
        assert False, 'opt.encoder.name is wrong!'
    return encoder


def feature_extract(opt, encoder, trainloader):
    fs = []
    ls = []
    encoder = encoder.to(opt.device)
    if opt.ddp:
        encoder = torch.nn.DataParallel(encoder)
    with torch.no_grad():
        encoder.eval()
        for idx, data in enumerate(trainloader):
            img, l, _ = data
            f = encoder(img.to(opt.device)).cpu().data
            fs.append(f)
            ls.append(l)
            print('\r\t', idx, '/', len(trainloader), end='' if idx + 1 < len(trainloader) else '\n', flush=True)
    fs = torch.cat(fs, dim=0)
    ls = torch.cat(ls, dim=0)
    return (fs, ls)


def import_fc(opt, encoder, trainloader=None):
    emb_size = encoder.emb_size
    if opt.fc.init_type == 'mean':
        fs, ls = feature_extract(opt, encoder, trainloader)
        name_ = opt.fc.type + '_mean'
        fc = metrics.__dict__[name_](opt, emb_size, fs, ls)
    else:
        fc = metrics.__dict__[opt.fc.type](opt, emb_size)
        if opt.fc.init_type == 'load':
            new_state_dict = load_normal(opt.fc.load_path)
            fc.load_state_dict(new_state_dict)

    return fc


def import_criterion(opt):
    return metrics.__dict__[opt.loss.type]()