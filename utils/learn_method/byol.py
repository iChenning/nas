import copy
import torch
from torch import nn
import torch.nn.functional as F

from utils.models.mlp import MLP


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(nn.Module):
    def __init__(self, encoder, emb_size=2048, hidden_size=4096, projection_size=256, beta=0.99):
        super(BYOL, self).__init__()
        self.beta = beta
        self.online_encoder = encoder
        self.online_projector = MLP(emb_size, hidden_size, projection_size)
        self.online_predictor = MLP(projection_size, hidden_size, projection_size)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

    @torch.no_grad()
    def _EMA_update_target(self):
        for param_o_e, param_t_e in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t_e.data = param_t_e.data * self.beta + param_o_e.data * (1. - self.beta)
        for param_o_p, param_t_p in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t_p.data = param_t_p.data * self.beta + param_o_p.data * (1. - self.beta)

    def forward(self, image_one, image_two):
        online_feat_one = self.online_encoder(image_one)
        online_proj_one = self.online_projector(online_feat_one)
        online_pred_one = self.online_predictor(online_proj_one)

        online_feat_two = self.online_encoder(image_two)
        online_proj_two = self.online_projector(online_feat_two)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            self._EMA_update_target()
            target_feat_one = self.target_encoder(image_one)
            target_proj_one = self.target_projector(target_feat_one)

            target_feat_two = self.target_encoder(image_two)
            target_proj_two = self.target_projector(target_feat_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        # print(loss.shape)
        return loss
