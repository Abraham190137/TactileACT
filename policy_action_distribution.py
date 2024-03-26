import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import IPython
e = IPython.embed

from detrvae import DETRVAE
from detr.models.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer
from detr.models.backbone import Backbone, Joiner, PositionEmbeddingLearned, PositionEmbeddingSine
from typing import Dict, List, Tuple
from visualization_utils import visualize_data, debug

class MyJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = [xs]
        pos = [self[1](xs).to(xs.dtype)]

        return out, pos

class ACTPolicy(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 position_embedding_type: str,
                 lr_backbone: float,
                 masks: bool,
                 backbone_type: str,
                 dilation: bool,
                 dropout: float,
                 nheads: int,
                 dim_feedforward: int,
                 num_enc_layers: int,
                 num_dec_layers: int,
                 pre_norm: bool,
                 num_queries: int,
                 camera_names,
                 z_dimension: int,
                 lr: float,
                 weight_decay: float,
                 kl_weight: float,
                 pretrained_backbones = None,
                 cam_backbone_mapping = None,
                 ):
        
        super().__init__()

        
        if cam_backbone_mapping is None:
            cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
            num_backbones = 1
        else:
            num_backbones = len(set(cam_backbone_mapping.values()))

        # if pretrained_backbones is not None:
        #     assert len(pretrained_backbones) == num_backbones
        if pretrained_backbones is not None:
            num_backbones = len(pretrained_backbones)

        # build model:
        # Build backbones:
        backbones = []
        for i in range(num_backbones):
            N_steps = hidden_dim // 2
            if position_embedding_type in ('v2', 'sine'):
                # TODO find a better way of exposing other arguments
                position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
            elif position_embedding_type in ('v3', 'learned'):
                position_embedding = PositionEmbeddingLearned(N_steps)
            else:
                raise ValueError(f"not supported {position_embedding_type}")
            
            train_backbone = lr_backbone > 0

            if pretrained_backbones is None:
                backbone = Backbone(name=backbone_type, 
                                    train_backbone=train_backbone, 
                                    return_interm_layers=masks, 
                                    dilation=dilation)
                backbone_model = Joiner(backbone, position_embedding)
                backbone_model.num_channels = backbone.num_channels
            else:
                backbone = pretrained_backbones[i]
                backbone_model = MyJoiner(backbone, position_embedding)
                backbone_model.num_channels = 512 #resnet18
            
            backbones.append(backbone_model)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=False,
        )

        # build encoder
        activation = "relu"

        encoder_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward,
                                                dropout, activation, pre_norm)
        encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)

        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=num_queries,
            camera_names=camera_names,
            z_dimension=z_dimension,
            cam_backbone_mapping=cam_backbone_mapping,
        )

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))
        self.model.cuda()

        # build optimizer
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                    weight_decay=weight_decay)
        
        self.kl_weight = kl_weight
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos:torch.Tensor, images, actions=None, is_pad=None, z=None, ignore_latent=False):
        global debug
        env_state = None
        # move normalize to dataloader
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        if actions is not None: # training time
            # actions = actions[:, :self.model.num_queries]
            # is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, images, env_state, actions, is_pad, debug=debug.print, ignore_latent=ignore_latent)
            
            visualize_data([img[0] for img in images], qpos[0], a_hat[0], is_pad[0], actions[0])

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, images, env_state, z=z) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu:torch.Tensor, logvar:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
