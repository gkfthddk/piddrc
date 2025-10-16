import os
import argparse
if __name__== '__main__':
    args=None
    parser=argparse.ArgumentParser()
    parser.add_argument("--balance",type=int,default=None,help='eval epoch')
    parser.add_argument("--epoch",type=int,default=None,help='eval epoch')
    parser.add_argument("--num_point",type=int,default=2048,help='num_point')
    parser.add_argument("--depth",type=int,default=3,help='depth')
    parser.add_argument("--emb_dim",type=int,default=16,help='emb_dim')
    parser.add_argument("--encoder_dim",type=int,default=64,help='encoder_dim')
    parser.add_argument("--in_channel",type=int,default=5,help='in channel')
    parser.add_argument("--group_size",type=int,default=16,help='emb_dim')
    parser.add_argument("--num_group",type=int,default=64,help='emb_dim')
    parser.add_argument("--gpu",type=str,default='2',help='gpu')
    parser.add_argument("--name",type=str,default='mamba_id',help='gpu')
    parser.add_argument("--eval", action='store_true', help="set noise")
    parser.add_argument("--cand",type=str,default='pi0_line_sako,gamma_line_sako,pi+_line_sako,e-_line_sako',help='cand')

    args=parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import sys
import json
import yaml
import torch as th
assert th.cuda.is_available()
from collections import namedtuple
from torchinfo import summary
from torcheval.metrics import BinaryAUROC
from torcheval.metrics import MeanSquaredError
import subprocess
from typing import Optional
import numpy as np
import gc
import datetime
import h5py
from sklearn.utils import shuffle
import tqdm
import math
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
from functools import partial

#from timm.models.layers import trunc_normal_
#from timm.models.layers import DropPath
from timm.layers import trunc_normal_
from timm.layers import DropPath

from mamba_ssm import Mamba
from mamba_ssm import Mamba2

optuna = {'is_optuna' : False, 'trial' : None}

models = {}

th.autograd.set_detect_anomaly(True)
def check(x, name):
    if not th.isfinite(x).all():
        print(f"[NaN/Inf] at {name}")
        print(th.isfinite(x).all(dim=1))
        raise RuntimeError(f"Bad values in {name}")

import chamfer
class ChamferFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(th.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = th.sum(xyz1, dim=2).ne(0)
            non_zeros2 = th.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return th.mean(dist1) + th.mean(dist2)

class ChamferDistanceL2_split(th.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = th.sum(xyz1, dim=2).ne(0)
            non_zeros2 = th.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return th.mean(dist1), th.mean(dist2)

class ChamferDistanceL1(th.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = th.sum(xyz1, dim=2).ne(0)
            non_zeros2 = th.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = th.sqrt(dist1)
        dist2 = th.sqrt(dist2)
        return (th.mean(dist1) + th.mean(dist2))/2


def tanh01(x):
    x[:,:,:2].tanh_()
    return x


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data

class Block(th.nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=th.nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        
        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else th.nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (th.nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: th.Tensor, residual: Optional[th.Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(th.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class Encoder(th.nn.Module):
    def __init__(self, input_channel, encoder_channel,num_feature):
        super().__init__()
        self.input_channel = input_channel
        self.encoder_channel = encoder_channel
        self.num_feature=num_feature
        self.first_conv = th.nn.Sequential(
            th.nn.Conv1d(input_channel, self.num_feature, 1),
            th.nn.BatchNorm1d(self.num_feature),
            th.nn.ReLU(inplace=True),
            th.nn.Conv1d(self.num_feature,self.num_feature*2,1)
        )
        self.second_conv = th.nn.Sequential(
            th.nn.Conv1d(self.num_feature*4,self.num_feature*4,1),
            th.nn.BatchNorm1d(self.num_feature*4),
            th.nn.ReLU(inplace=True),
            th.nn.Conv1d(self.num_feature*4, self.encoder_channel, 1)
        )
    def forward(self, point_groups):

        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.input_channel)

        feature = self.first_conv(point_groups.transpose(2,1))
        feature_global = th.max(feature, dim=2, keepdim=True)[0]
        feature = th.cat([feature_global.expand(-1,-1,n).detach(), feature.detach()], dim=1)
        feature = self.second_conv(feature)
        feature_global = th.max(feature, dim=2, keepdim=False)[0]

        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(th.nn.Module):
    def __init__(self, input_channel,num_group, group_size):
        super().__init__()
        self.input_channel = input_channel
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self,xyz):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group) # batch, num_group, input_channel
        #center = fps(xyz, num_points//self.num_group)
        _, idx = self.knn(xyz,center)# _ , batch, num_group, group_size
        idx_base = th.arange(0, batch_size, device=xyz.device).view(-1,1,1) * num_points
        #print(xyz.shape,center.shape,idx.shape,_.shape,idx_base.shape,num_points,self.num_group)
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size,self.input_channel).contiguous()
        #neighborhood = neighborhood.view(batch_size, num_points//self.num_group, self.group_size,self.input_channel).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, th.nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                th.nn.init.zeros_(module.bias)
    elif isinstance(module, th.nn.Embedding):
        th.nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                th.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with th.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def create_block(
        layer_module,
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device='cuda',
        dtype=None,
    ):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
      
    mixer_cls = partial(layer_module, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        th.nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(th.nn.Module):
    def __init__(
            self,
            layer_module,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device='cuda',
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = th.nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = th.nn.ModuleList(
            [
                create_block(
                    layer_module,
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (th.nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else th.nn.Identity()
        self.drop_out_in_block = th.nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else th.nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, masked=None, inference_params=None):
    #def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        if(not masked is None):
            hidden_states=hidden_states*masked.unsqueeze(-1).to(hidden_states.dtype)
        residual = None
        #hidden_states = hidden_states + pos
        for layer in self.layers:
            #print("hidden_state",layer,hidden_states.shape)
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if(not masked is None):
            hidden_states=hidden_states*masked.unsqueeze(-1).to(hidden_states.dtype)

        return hidden_states

class UncerntaintyLoss(th.nn.Module):
    def __init__(self, num_task):
        super().__init__()
        self.log_sigma = th.nn.Parameter(th.zeros(1+num_task))
    def forward(self,losses):
        precision_weight=th.exp(-self.log_sigma)
        loss_mean=th.mean(th.stack(losses)).detach()
        loss_std=th.std(th.stack(losses)).detach()
        #losses = [(loss - loss_mean) / (loss_std + 1e-6) for loss in losses]
        #losses = [loss / (loss_std + 1e-6) for loss in losses]
        losses = [loss / (loss_mean + 1e-6) for loss in losses]
        #weighted_loss=[precision_weight[i]*losses[i] + self.log_sigma*0.01 for i in range(len(self.log_sigma))]
        weighted_loss=[0.5*precision_weight[i]*losses[i] + self.log_sigma*0.01 for i in range(len(self.log_sigma))]
        total_loss=th.stack(weighted_loss).sum()
        #total_loss=th.stack(losses).sum()

        reg_term=0.01*th.sum(self.log_sigma**2)

        return total_loss+reg_term
class LogCoshLoss(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_t,y_prime_t):
        ey_t=y_t-y_prime_t
        return th.mean(th.log(th.cosh(ey_t+1e-12)))

class MLP(th.nn.Module):
    def __init__(self, input_channel=2, feature_dim=[256,512], dropout=0.,model_config=None, **kwargs):
        super(MLP, self).__init__()
        self.input_channel=input_channel
        self.last_dim = model_config.last_dim
        self.cls_dim = model_config.cls_dim
        self.reg_dim = model_config.reg_dim

        self.linear_dims=[self.input_channel]+feature_dim+[self.last_dim]
        self.node=[]
        for i in range(len(self.linear_dims)-1):
            dim1=self.linear_dims[i]
            dim2=self.linear_dims[i+1]
            self.node.append(
                th.nn.Sequential(
                    th.nn.Linear(dim1,dim2),
                    th.nn.BatchNorm1d(dim2),
                    th.nn.ReLU(inplace=True),
                    th.nn.Dropout(dropout),
                )
            )
        self.block=th.nn.Sequential(*self.node)

        self.cls_head = th.nn.Sequential(
                th.nn.Linear(self.linear_dims[-1], self.cls_dim),
                )
        self.reg_head=th.nn.Sequential(
              th.nn.Linear(self.linear_dims[-1], self.reg_dim*2,device="cuda"),
              )
        with th.no_grad():
            self.reg_head[0].bias[self.reg_dim:]=0.0
        self.return_ret=False

    def forward(self,x1):
        if(len(x1.shape)>2):
            x1=x1.squeeze(1)
        ret=self.block(x1)
        if(self.return_ret):
            return ret
        cls=self.cls_head(ret)
        reg=self.reg_head(ret)
        mu,log_sigma=reg.chunk(2,dim=-1)

        return cls,mu,log_sigma
class CrossAttention(th.nn.Module):
    def __init__(self,dim_q,dim_kv,dim_out,num_head=4):
        super().__init__()
        self.attn=th.nn.MultiheadAttention(embed_dim=dim_out,kdim=dim_kv,vdim=dim_kv,num_heads=num_head,batch_first=True)
        self.query_proj=th.nn.Linear(dim_q,dim_out)
        self.out_proj=th.nn.Linear(dim_out,dim_out)
        self.ln_q=th.nn.LayerNorm(dim_out)
        self.ln_kv=th.nn.LayerNorm(dim_kv)
    def forward(self,query,keyval,mask=None):
        q=self.ln_q(self.query_proj(query))
        keyval=self.ln_kv(keyval)
        attn_out,_=self.attn(q,keyval,keyval,key_padding_mask=mask)
        return self.out_proj(attn_out)

class MODE3(th.nn.Module):
    def __init__(self, config, **kwargs):
        super(MODE3, self).__init__()
        self.config = config
        self.net0 = PointMODE(config.model0,**kwargs)
        self.net1 = PointMODE(config.model,**kwargs)
        self.net2 = MLP(model_config=config.model0)

        self.net0.return_ret=True
        self.net1.return_ret=True
        self.net2.return_ret=True

        self.cross0=CrossAttention(dim_q=self.net2.last_dim,dim_kv=self.net0.trans_dim,dim_out=self.net2.last_dim)
        self.cross1=CrossAttention(dim_q=self.net2.last_dim,dim_kv=self.net1.trans_dim,dim_out=self.net2.last_dim)
        self.cross=config.model.cross
        self.film=config.model.film

        channel3=self.net0.last_dim+self.net1.last_dim+self.net2.last_dim
        self.net3 = MLP(input_channel=channel3,feature_dim=[],model_config=config.model)
        
    def forward(self,x0,x1,x2):
        mask0 = (x0[:,:,0] == 0)
        mask1 = (x1[:,:,0] == 0)
        ret2_global=self.net2(x2)
        if(self.cross):
            _,seq0=self.net0(x0)
            _,seq1=self.net1(x1)
            ret2_unsqueezed=ret2_global.unsqueeze(1)
            ret2_global=ret2_unsqueezed+self.cross0(ret2_unsqueezed,seq0,mask0)+self.cross1(ret2_unsqueezed,seq1,mask1)
        ret2_global=ret2_global.squeeze(1)
        if(self.film):
            ret0,_=self.net0(x0,ret2_global)
            ret1,_=self.net1(x1,ret2_global)
        else:
            ret0,_=self.net0(x0)
            ret1,_=self.net1(x1)
        
        ret=th.cat([ret0,ret1,ret2_global],dim=1)
        return self.net3(ret)

def masked_max(h, mask, dim=1):
    """
    h:    [B, T, D] tensor from MixerModel/Mamba stack
    mask: [B, T]    bool (True=valid, False=pad)
    dim:  reduction dimension (time/points)

    returns: pooled [B, D]
    """
    m = mask.unsqueeze(-1)                      # [B, T, 1]
    # choose a very negative fill value that works for the dtype
    if h.dtype in (th.float16, th.bfloat16):
        neg_inf = th.finfo(th.float32).min  # safer than fp16 min
    else:
        neg_inf = th.finfo(h.dtype).min

    h_masked = h.masked_fill(~m, neg_inf)       # pads -> -inf (or very negative)
    pooled = h_masked.amax(dim=dim)       # [B, D]

    # guard sequences that are entirely padded (optional safety)
    all_pad = (mask.sum(dim=dim) == 0).unsqueeze(-1)  # [B,1]
    pooled = th.where(all_pad, th.zeros_like(pooled), pooled)
    return pooled

def masked_mean(h, masked, dim=1):
    """
    h:    [B, T, D] tensor from MixerModel/Mamba stack
    masked: [B, T]    bool (True=valid, False=pad)
    dim:  reduction dimension (time/points)

    returns: pooled [B, D]
    """
    masked = masked.unsqueeze(-1) # [B, T, 1]
    summed=(h*masked).sum(dim=dim)
    count=masked.sum(dim=dim)
    mean=summed/count.clamp(min=1).float()
    return mean

class PMA(th.nn.Module):
    def __init__(self,dim_in,pma_dim=1,dim_out=None,num_head=4):
        super().__init__()
        self.pma_dim=pma_dim
        self.seed_vector=th.nn.Parameter(th.randn(1,pma_dim,dim_in))
        self.attn=th.nn.MultiheadAttention(embed_dim=dim_in,num_heads=num_head,batch_first=True)
        if dim_out is None:
            self.out_proj=th.nn.Identity()
        else:
            self.out_proj=th.nn.Linear(dim_in,dim_out)
    def forward(self,x,mask=None):
        B=x.size(0)
        seed=self.seed_vector.expand(B,-1,-1)
        if mask is not None:
            attn_mask=mask.bool()
        else:
            attn_mask=None

        pooled,_=self.attn(seed,x,x,key_padding_mask=attn_mask)
        return self.out_proj(pooled)

class FiLM(th.nn.Module):
    def __init__(self,feat_dim,cond_dim,hidden_dim=32,num_layer=2):
        super().__init__()
        def make_mlp(in_dim, out_dim):
            layers = []
            for _ in range(num_layer - 1):
                layers.append(th.nn.Linear(in_dim, hidden_dim))
                #layers.append(th.nn.BatchNorm1d(hidden_dim))
                layers.append(th.nn.LayerNorm(hidden_dim))
                layers.append(th.nn.ReLU())
                in_dim = hidden_dim
            layers.append(th.nn.Linear(hidden_dim, out_dim))
            return th.nn.Sequential(*layers)

        self.gamma = make_mlp(cond_dim, feat_dim)
        self.beta = make_mlp(cond_dim, feat_dim)
        self.gamma[-1].weight.data.zero_()
        self.gamma[-1].bias.data.fill_(1.0)
        self.beta[-1].weight.data.zero_()
        self.beta[-1].bias.data.zero_()

    def forward(self,x,cond):
        g=self.gamma(cond).unsqueeze(1)
        b=self.beta(cond).unsqueeze(1)
        return g*x+b

class PointMODE(th.nn.Module):
    def __init__(self, model_config, **kwargs):
        super(PointMODE, self).__init__()
        self.model_config = model_config

        self.name = model_config.NAME
        self.trans_dim = model_config.trans_dim
        self.depth = model_config.depth
        self.cls_dim = model_config.cls_dim
        self.reg_dim = model_config.reg_dim

        self.group_size = model_config.group_size
        self.num_group = model_config.num_group
        self.encoder_dim = model_config.encoder_dim
        self.input_channel = model_config.in_channel
        self.emb_dim = model_config.emb_dim
        if(self.emb_dim==0):
            self.emb_dim=None

        self.pma_dim = model_config.pma_dim
        self.fine_dim = model_config.fine_dim
        self.last_dim = model_config.last_dim
        self.film0 = FiLM(self.input_channel,model_config.last_dim)
        #self.film1 = FiLM(self.trans_dim,model_config.last_dim)
        self.model_args=model_config._asdict()
        self.weight = th.nn.Parameter(th.ones(1+self.reg_dim)).to("cuda")

        if("Mamba" in self.name):
            self.group_divider = Group(input_channel=self.input_channel, num_group=self.num_group, group_size=self.group_size)
            self.encoder = Encoder(input_channel=self.input_channel, encoder_channel=self.encoder_dim, num_feature=self.encoder_dim)
        if("Simba" in self.name or "MLP" in self.name or "Transformer" in self.name):
            self.encoder = th.nn.Sequential(
                th.nn.Linear(self.input_channel,self.encoder_dim),
                th.nn.Linear(self.encoder_dim,self.trans_dim),
                )
            if(self.pma_dim>0):
                self.pma=PMA(self.trans_dim,pma_dim=self.pma_dim)
            else:
                self.pma=None

        self.use_cls_token = False if not hasattr(self.model_args, "use_cls_token") else self.model_args['use_cls_token']
        self.drop_path = 0. if not hasattr(self.model_args, "drop_path") else self.model_args['drop_path']
        self.rms_norm = False if not hasattr(self.model_args, "rms_norm") else self.model_args['rms_norm']
        self.drop_out_in_block = 0. if not hasattr(self.model_args, "drop_out_in_block") else self.model_args['drop_out_in_block']

        if self.use_cls_token:
            self.cls_token = th.nn.Parameter(th.zeros(1, 1, self.trans_dim))
            self.cls_pos = th.nn.Parameter(th.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        if("Mamba"==self.name or "Simba"==self.name):
            if(not self.emb_dim is None):
                self.pos_embed = th.nn.Sequential(
                    th.nn.Linear(self.input_channel, self.emb_dim),
                    th.nn.GELU(),
                    th.nn.Linear(self.emb_dim, self.trans_dim)
                )
            layer_module=Mamba
        if("Mamba2"==self.name):
            layer_module=Mamba2
        if("Mamba" in self.name or "Simba" in self.name):
            self.blocks = MixerModel(layer_module,
                                 d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)
        if("Transformer" in self.name):
            self.encoder_layer = th.nn.TransformerEncoderLayer(d_model=self.trans_dim,nhead=4,batch_first=True,norm_first=True,dim_feedforward=256,)
            self.blocks = th.nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=4,enable_nested_tensor=False)

        if(self.name=="MLP"):
            self.blocks = th.nn.Sequential(
                th.nn.Linear(self.trans_dim,self.trans_dim),
                th.nn.LayerNorm(self.trans_dim),
                th.nn.ReLU(inplace=True),
                th.nn.Dropout(0.0),
            )
        if(self.name=="Conv"):
            self.blocks = th.nn.Sequential(
                th.nn.Conv1d(self.trans_dim*self.input_channel,self.trans_dim,1),
                #th.nn.BatchNorm1d(self.trans_dim),
                th.nn.ReLU(inplace=True),
                th.nn.Conv1d(self.trans_dim,self.trans_dim,1),
                #th.nn.BatchNorm1d(self.trans_dim),
                th.nn.ReLU(inplace=True),
            )

        self.layer_norm = th.nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = self.pma_dim+2
        if self.use_cls_token:
            self.HEAD_CHANEL += 1
        self.head_finetune = th.nn.Sequential(
            th.nn.Linear(self.trans_dim * self.HEAD_CHANEL, self.fine_dim),
            #th.nn.BatchNorm1d(self.fine_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout(0.1),
            th.nn.Linear(self.fine_dim, self.fine_dim),
            #th.nn.BatchNorm1d(self.fine_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout(0.1),
        )
        self.head_tune = th.nn.Linear(self.fine_dim, self.last_dim)
        self.cls_head = th.nn.Sequential(
                th.nn.Linear(self.last_dim, self.last_dim),
                th.nn.Linear(self.last_dim, self.cls_dim)
                )
        #= th.nn.Linear(self.last_dim, self.reg_dim)
        self.reg_head0 = None
        self.reg_head1 = None
        if self.reg_dim>0:
          self.reg_head0=th.nn.Sequential(
                  th.nn.Linear(self.last_dim, self.last_dim,device="cuda"),
                  th.nn.Linear(self.last_dim, 1,device="cuda"),
                  )
        if self.reg_dim>1:
          self.reg_head1=th.nn.Sequential(
                  th.nn.Linear(self.last_dim, self.last_dim,device="cuda"),
                  th.nn.Linear(self.last_dim, 1,device="cuda"),
                  )

        self.build_loss_func()
        self.drop_out = th.nn.Dropout(model_config['drop_out']) if "drop_out" in model_config else th.nn.Dropout(0)
        self.return_ret=False

    def build_loss_func(self):
        self.loss_ce = th.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_mse = th.nn.MSELoss()
        #self.loss_logcosh = th.nn.LogCoshLoss()
        self.loss_sml1 = th.nn.SmoothL1Loss(beta=0.1)


    def get_losses(self, output, target,norm=False):
        losses= [self.loss_ce(output[0], target[0])]
        #loss2 = self.loss_sml1(output[1],target[1])
        if(len(target[1][0])>0):
            for i in range(len(target[1][0])):
                losses.append(self.loss_sml1(output[i+1].ravel(),target[1][:,i]))
        tm=[]
        if norm:
            for i in range(len(losses)):
                tm.append(th.max(th.abs(output[i])))
            tm_sum=sum(tm)
            for i in range(len(losses)):
                losses[i]=losses[i]*tm_sum/(tm[i]+1)
        return losses

    def gradnorm_lossnbackward(self,output,target,alpha=0.01):
        losses=self.get_losses(output,target)
        weighted_loss=[self.weight[i].detach()*losses[i] for i in range(len(self.weight))]
        total_loss=sum(weighted_loss)
        gw=[]
        for i in range(len(self.weight)):
            dl=th.autograd.grad(self.weight[i]*losses[i],self.head_tune.parameters(),retain_graph=True,create_graph=True)
            grad_norm = th.norm(th.stack([g.norm(2) for g in dl if g is not None]))
            #gw.append(th.norm(dl))
            gw.append(grad_norm)
        total_loss.backward(retain_graph=True)
        mean_grad_norm = sum(gw)/len(gw)
        with th.no_grad():
            self.weight/=self.weight.sum()
        for i in range(len(self.weight)):
            self.weight.data[i]*=(gw[i]/mean_grad_norm).pow(alpha)
            #if(i==0):self.weight.data[i]=max(self.weight[i].item(),0.05)
            self.weight.data[i]=max(self.weight[i].item(),0.002)
        with th.no_grad():
            self.weight/=self.weight.sum()
            self.weight*=len(self.weight)

        return total_loss

    def get_loss(self, output, target):
        losses=self.get_losses(output,target)
        total_loss=th.stack(losses).sum()
        return total_loss
    def set_loss(self):
        self.total_loss=th.tensor(0.0,device='cuda')
    def pin_loss(self,output,target):
        self.total_loss.zero_()
        self.total_loss.add_(self.loss_ce(output[0], target[0]))
        if(len(target[1][0])>0):
            for i in range(len(target[1][0])):
                self.total_loss.add_(self.loss_sml1(output[i+1][:,0],target[1][:,i]))

    def _init_weights(self, m):
        if isinstance(m, th.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, th.nn.Linear) and m.bias is not None:
                th.nn.init.constant_(m.bias, 0)
        elif isinstance(m, th.nn.LayerNorm):
            th.nn.init.constant_(m.bias, 0)
            th.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, th.nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                th.nn.init.constant_(m.bias, 0)

    def forward(self, pts,cond=None):
        if("Mamba" in self.name):
            neighborhood, center = self.group_divider(pts)
            group_input_tokens = self.encoder(neighborhood)  # B G N
            if(not self.emb_dim is None):
                pos = self.pos_embed(center)
            # reordering strategy
            center_bond=[]
            group_input_tokens_bond=[]
            pos_bond=[]
            for i in range(self.input_channel):
                center_bond.append(center[:, :, i].argsort(dim=-1)[:, :, None])
                group_input_tokens_bond.append(group_input_tokens.gather(dim=1, index=th.tile(center_bond[i], (1, 1, group_input_tokens.shape[-1]))))
                if(not self.emb_dim is None):
                    pos_bond.append(pos.gather(dim=1, index=th.tile(center_bond[i], (1, 1, pos.shape[-1]))))
            group_input_tokens = th.cat(group_input_tokens_bond, dim=1)
            x = group_input_tokens
            # transformer
            x = self.drop_out(x)
            if(not self.emb_dim is None):
                pos = th.cat(pos_bond, dim=1)
                x = self.blocks(x+pos)
            else:
                x = self.blocks(x)
            x = self.layer_norm(x)
            concat_f = x[:, :].mean(1)
            ret = self.head_finetune(concat_f)
        elif("Simba" in self.name or "MLP" in self.name or "Transformer" in self.name):
            masked = (pts[:,:,0] != 0)
            #mask=(pts != 0).any(dim=-1)
            if(cond is not None):
                pts=self.film0(pts,cond)
            input_tokens = self.encoder(pts)
            if(cond is not None):
                input_tokens=input_tokens#+self.film1(input_tokens,cond)
            if("Simba" in self.name):
                x = self.blocks(input_tokens,masked)
            elif("MLP" in self.name):
                x = self.blocks(input_tokens)
            elif("Transformer" in self.name):
                x = self.blocks(input_tokens,src_key_padding_mask=~masked)
            x = self.layer_norm(x)#need to test layernormalization if not you can remove lyaernormalization.
            #concat_f = masked_max(x, masked)
            if(self.pma_dim>0 and self.pma):
                concat_f = th.cat([self.pma(x, ~masked).reshape(x.size(0),-1),masked_max(x,masked),masked_mean(x,masked)],dim=-1)
            else:
                concat_f = th.cat([masked_max(x,masked),masked_mean(x,masked)],dim=-1)
            ret = self.head_finetune(concat_f)

        ret = self.head_tune(ret)
        if(self.return_ret==True):
            return ret,x
        ret1 = self.cls_head(ret)
        if(self.reg_head0 is not None):
          red0=None
          red1=None
          if(self.reg_head0 is not None):
              red0=self.reg_head0(ret)
          if(self.reg_head1 is not None):
              red1=self.reg_head1(ret)
          return ret1,red0,red1
        else:
          return ret1

from collections import namedtuple

def re_namedtuple(pd):
  if isinstance(pd,dict):
    for key in pd:
      if isinstance(pd[key],dict):
        pd[key] = re_namedtuple(pd[key])
    pod = namedtuple('cfg',pd.keys())
    bobo = pod(**pd)

  return bobo
