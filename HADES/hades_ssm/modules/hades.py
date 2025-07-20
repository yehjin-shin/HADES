# Copyright (c) 2024, Tri Dao, Albert Gu.
# modified for HADES
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from ..ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from ..ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from ..distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from ..distributed.distributed_utils import all_reduce, reduce_scatter

from ..ops.triton.ssd_combined import mamba_chunk_scan_combined
from ..ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class HADES(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        num_filters=None, # Number of selected filter, H
        shared_filters=None, # Number of shared filter, S
        gamma=None, # Control the strength of spectral bias
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device='cuda',
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        # configs for heads
        self.num_filters = num_filters # number of selected filters, H
        self.headdim = headdim
        self.total_filters = ((self.expand * self.d_model) // self.world_size) // self.headdim # number of total filters, M
        self.shared_filters = shared_filters # number of shared filters, S
        self.select_filters = self.total_filters - self.shared_filters # number of candidate filters, M - S
        # for spectral residual bias
        self.gamma = torch.tensor(gamma, **factory_kwargs)
        self.shared_ids = torch.arange(self.shared_filters, device=device) + self.select_filters # for router id selection
        # basic HADES configs
        self.d_inner = (self.num_filters * self.headdim) // self.world_size
        self.d_ssm = (self.num_filters * self.headdim) // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.total_filters #+ self.select_filters
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

            
        if self.process_group is None:
            self.h_proj = nn.Linear(self.d_model+self.total_filters, self.select_filters+self.num_filters-self.shared_filters, bias=bias, **factory_kwargs)
        else:
            self.h_proj = ColumnParallelLinear(self.d_model+self.total_filters, (self.select_filters+self.num_filters-self.shared_filters) * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)


        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(1, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True
        
        # Initialize gamma
        self.gamma = nn.Parameter(self.gamma.repeat(self.num_filters))
        self.gamma._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.num_filters, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.num_filters, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def load_balance_loss(self, x):
        eps = 1e-10
        # if only num_filters = 1
        if x.shape[0] == 1:
            return torch.tensor(0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def diversity_loss(self, outputs):
        outputs = F.normalize(outputs, p=2, dim=-1)
        similarity = torch.einsum('bslh,bsmh->bslm', outputs, outputs)
        off_diagonal = similarity - torch.eye(similarity.size(-1), device=similarity.device)
        loss = (off_diagonal ** 2).mean()
        return loss

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, **kwargs):
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state, cumsum_state = self._get_states_from_cache(inference_params, inference_batch) # cumsum_state for \mu_t
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _, _ = self.step(u, conv_state, ssm_state, cumsum_state, inference_params.seqlen_offset)
                return out, (0, 0)
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)

        zxbc, dt = torch.split(zxbcdt, [zxbcdt.shape[-1] - self.total_filters, self.total_filters], dim=-1)
        ## spectral bias generation : x_t - mu_t
        spectral_residual = u - torch.cumsum(u, dim=1) / torch.arange(1, seqlen+1, device=u.device).view(1, -1, 1)
        udt = torch.cat([spectral_residual, dt], dim=-1)

        hb = self.h_proj(udt)
        h, spectral_bias = torch.split(hb, [self.select_filters, hb.shape[-1]-self.select_filters], dim=-1)

        load_balance_loss = self.load_balance_loss(h)
        
        ## Router
        _, select_ids = torch.topk(h, self.num_filters - self.shared_filters, dim=-1)
        shared_ids = self.shared_ids.repeat(select_ids.shape[0], select_ids.shape[1], 1)
        ids = torch.cat([select_ids, shared_ids], dim=-1)
        dt = torch.gather(dt, 2, ids)

        ## Final delta HADES
        spectral_bias = self.gamma * torch.concat([spectral_bias, torch.zeros_like(shared_ids)], dim=-1)
        dt = dt + self.dt_bias + spectral_bias
        
        zxbcdt = torch.concat([zxbc, dt], dim=-1)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (num_filters) or (d_inner, d_state)

        expert_norm_weight = self.norm.weight

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                None,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=expert_norm_weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=None,
                outproj_bias=None,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            out = rearrange(out, "b l (h p) -> b l h p", h=self.num_filters)
            self.visualize_out = out
            # y: (b, l, h, p)
            # Compute diversity loss among selected experts (on outputs)
            diversity_loss = self.diversity_loss(out)
            # Combine expert outputs for out_proj
            out = rearrange(out, "b l h p -> b l (h p)", h=self.num_filters)            
            out = self.out_proj(out)

            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.num_filters) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.num_filters],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.dconv - 1):]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=None, # expert_dt_bias
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)

            if cumsum_state is not None: # update cumulative sum for mean calculation
                cumsum_state.copy_(u.sum(axis=1))

            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")

            y = rearrange(y, "b l (h p) -> b l h p", h=self.num_filters)
            diversity_loss = self.diversity_loss(y).to(y.device) # diversity loss calculation
            y = rearrange(y, "b l h p -> b l (h p)")
            out = self.out_proj(y)

        return out, (load_balance_loss, diversity_loss)

    def step(self, hidden_states, conv_state, ssm_state, cumsum_state, t_pos):
        # t_pos : current position of sequence
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        u = hidden_states.squeeze(1)
        zxbcdt = self.in_proj(u)  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.total_filters) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.total_filters],
            dim=-1
        )
        # calculation of spectral bias
        spectral_residual = u - (cumsum_state/(t_pos-1))
        udt = torch.cat([spectral_residual, dt], dim=-1)
        hb = self.h_proj(udt)
        h, spectral_bias = torch.split(hb, [self.select_filters, hb.shape[-1]-self.select_filters], dim=-1)

        # router selection
        _, select_ids = torch.topk(h, self.num_filters - self.shared_filters, dim=-1)
        shared_ids = self.shared_ids.reshape(1, -1).repeat(select_ids.shape[0], 1)
        ids = torch.cat([select_ids, shared_ids], dim=-1)
        dt = torch.gather(dt, -1, ids)

        # adding bias
        spectral_bias = self.gamma * torch.concat([spectral_bias, torch.zeros_like(shared_ids)], dim=-1)
        dt = dt + spectral_bias # original dt_bias is separately added
        cumsum_state.copy_(cumsum_state + u) # update cumsum_state
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (num_filters,)

        expert_dt_bias = self.dt_bias.repeat(self.num_filters) # fixed dt_bias for each filters

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + expert_dt_bias.to(dtype=dt.dtype))  # (batch, num_filters)
            dA = torch.exp(dt * A)  # (batch, num_filters)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(expert_dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state, cumsum_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.num_filters, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        # initialize cumsum_state cache
        cumsum_state = torch.zeros(
            batch_size, self.d_model, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state, cumsum_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.num_filters,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            # initialized cumsum_state cache
            cumsum_state = torch.zeros(
                batch_size, 
                self.d_model, 
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state, cumsum_state)
        else:
            conv_state, ssm_state, cumsum_state = inference_params.key_value_memory_dict[self.layer_idx]
            
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
                cumsum_state.zero_()
        return conv_state, ssm_state, cumsum_state
