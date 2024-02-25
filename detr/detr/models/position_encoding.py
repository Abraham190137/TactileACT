# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

# from ..util.misc import NestedTensor

import IPython
e = IPython.embed

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):#: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding

if __name__ == '__main__':
    # position_embedding = PositionEmbeddingSine(512//2)
    import matplotlib.pyplot as plt
    # pos = position_embedding(torch.zeros(1, 512, 10, 10))
    # print(pos.shape)
    # plt.imshow(pos[0].flatten(1).cpu().numpy().T)
    # plt.show()

    import numpy as np

    def positional_encoding(height, width, d_model):
        """
        Generates positional encodings for a given feature map size.
        
        Args:
        - height (int): Height of the feature map.
        - width (int): Width of the feature map.
        - d_model (int): Dimensionality of the model.
        
        Returns:
        - np.array: Positional encodings of shape (height * width, d_model).
        """
        pos_enc = np.zeros((height, width, d_model))
        for pos_row in range(height):
            for pos_col in range(width):
                for i in range(0, d_model, 2):
                    pos_enc[pos_row, pos_col, i] = np.sin(pos_row / (10000 ** ((2 * i)/d_model)))
                    pos_enc[pos_row, pos_col, i + 1] = np.cos(pos_col / (10000 ** ((2 * (i + 1))/d_model)))
                    
        pos_enc = pos_enc.reshape(-1, d_model)
        return pos_enc

    # Assuming h x w x 512 feature map
    h, w, d_model = 8, 8, 512
    n_tokens = h * w

    # Generate positional encoding
    pos_encodings = positional_encoding(h, w, d_model)

    print("Shape of positional encodings:", pos_encodings.shape)
    plt.imshow(pos_encodings)
    plt.show()




