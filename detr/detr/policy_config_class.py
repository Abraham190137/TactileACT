from typing import List, Optional, Union, Dict, Any
import os

class PolicyConfig:
    def __init__(self, 
                 lr: float,
                 num_queries: int,
                 kl_weight: float,
                 hidden_dim: int,
                 dim_feedforward: int,
                 lr_backbone: float,
                 backbone: str,
                 enc_layers: int,
                 dec_layers: int,
                 nheads: int,
                 camera_names: List[str],
                 state_dim: int):
        
        self.lr = lr
        self.num_queries = num_queries
        self.kl_weight = kl_weight
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.lr_backbone = lr_backbone
        self.backbone = backbone
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.nheads = nheads
        self.camera_names = camera_names
        self.state_dim = state_dim
        

class ModelConfiguration():
    def __init__(self,
                 is_eval: bool,
                 save_dir: str,
                 model_name: str,
                 policy_class: str,
                 batch_size_train: int,
                 batch_size_val: int,
                 num_epochs: int,
                 learning_rate: float,
                 chunk_size: int,
                 kl_weight: float,
                 start_kl_epoch: int,
                 kl_scale_epochs: int,
                 hidden_dim: int,
                 dim_feedforward: int,
                 lr: float,
                 lr_backbone: float,
                 backbone: str,
                 enc_layers: int,
                 dec_layers: int,
                 nheads: int,
                 state_dim: int,
                 seed: int,
                 temporal_agg: bool,
                 ckpt_name: str,
                 gpu: int,
                 onscreen_render: bool,
                 ):
        
        self.is_eval = is_eval
        self.save_dir = save_dir
        self.model_name = model_name
        self.policy_class = policy_class
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.chunk_size = chunk_size
        self.kl_weight = kl_weight
        self.start_kl_epoch = start_kl_epoch
        self.kl_scale_epochs = kl_scale_epochs
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.backbone = backbone
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.nheads = nheads
        self.state_dim = state_dim
        self.seed = seed
        self.temporal_agg = temporal_agg
        self.ckpt_name = ckpt_name
        self.gpu = gpu
        self.onscreen_render = onscreen_render                       

        self.ckpt_dir: str = os.path.join(self.save_dir, self.model_name)
        self.dataset_dir: str = os.path.join(self.save_dir, 'data')

        assert os.path.exists(self.save_dir), f'{self.save_dir} does not exist. Please select a valid directory.'

        # read the meta_data folder:
        with open(os.path.join(self.save_dir, 'meta_data.json'), 'r') as f:
            meta_data: Dict[str, Any] = json.load(f)
        self.task_name: str = meta_data['task_name']
        self.num_episodes: int = meta_data['num_episodes']
        self.episode_len: int = meta_data['episode_length']
        self.camera_names: List[str] = meta_data['camera_names']
        self.is_sim: bool = meta_data['is_sim']

    @property 
    def policy_config(self) -> PolicyConfig:
        return PolicyConfig(lr=self.lr,
                            num_queries = self.chunk_size,
                            kl_weight= self.kl_weight,
                            hidden_dim=self.hidden_dim,
                            dim_feedforward=self.dim_feedforward,
                            lr_backbone=self.lr_backbone,
                            backbone=self.backbone,
                            enc_layers=self.enc_layers,
                            dec_layers=self.dec_layers,
                            nheads=self.nheads,
                            camera_names=self.camera_names,
                            state_dim=self.state_dim)