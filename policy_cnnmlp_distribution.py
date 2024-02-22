import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import IPython
e = IPython.embed

from detr.models.backbone import Backbone, Joiner, PositionEmbeddingLearned, PositionEmbeddingSine   
from cnnmpl import CNNMLP


class CNNMLPPolicy(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 lr_backbone: float,
                 masks: bool,
                 backbone_type: str,
                 dilation: bool,
                 camera_names,
                 lr: float,
                 weight_decay: float,
                 ):
        super().__init__()

        backbones = []
        for _ in camera_names:
            N_steps = hidden_dim // 2
            if position_embedding in ('v2', 'sine'):
                # TODO find a better way of exposing other arguments
                position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
            elif position_embedding in ('v3', 'learned'):
                position_embedding = PositionEmbeddingLearned(N_steps)
            else:
                raise ValueError(f"not supported {position_embedding}")
            
            train_backbone = lr_backbone > 0
            return_interm_layers = masks
            backbone = Backbone(backbone_type, train_backbone, return_interm_layers, dilation)
            backbone_model = Joiner(backbone, position_embedding)
            backbone_model.num_channels = backbone.num_channels

            backbones.append(backbone_model)

        model = CNNMLP(
            backbones,
            state_dim=state_dim,
            camera_names=camera_names,
        )

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))
        model.cuda()

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                    weight_decay=weight_decay)

        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        # move normalize to dataloader
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

