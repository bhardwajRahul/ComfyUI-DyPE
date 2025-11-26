import torch
import torch.nn as nn
import math
import types
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling
from .rope import get_1d_dype_yarn_pos_embed, get_1d_yarn_pos_embed, get_1d_ntk_pos_embed

class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', yarn_alt_scaling: bool = False, dype: bool = True, dype_scale: float = 2.0, dype_exponent: float = 2.0):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.yarn_alt_scaling = yarn_alt_scaling
        self.dype = True if method == 'vision_yarn' else (dype if method != 'base' else False)
        self.dype_scale = dype_scale
        self.dype_exponent = dype_exponent
        self.current_timestep = 1.0
        self.base_resolution = 1024
        self.base_patches = (self.base_resolution // 8) // 2

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep


    def _forward_vision_yarn(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        New Method: DyPE Vision YaRN (Decoupled + Quadratic Aggressive)
        """
        n_axes = pos.shape[-1]
        emb_parts = []
        
        # global Scale
        current_patches_h = int(pos[..., 1].max().item() + 1)
        current_patches_w = int(pos[..., 2].max().item() + 1)
        max_current_patches = max(current_patches_h, current_patches_w)
        
        scale_global = max(1.0, max_current_patches / self.base_patches)
            
        # dynamic MScale (Quadratic Decay)
        mscale_start = 0.1 * math.log(scale_global) + 1.0
        mscale_end = 1.0
        
        t_factor = math.pow(self.current_timestep, 2.0)
        current_mscale = mscale_end + (mscale_start - mscale_end) * t_factor

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            
            current_patches = int(axis_pos.max().item() + 1)
            
            common_kwargs = {
                'dim': axis_dim, 
                'pos': axis_pos, 
                'theta': self.theta, 
                'use_real': True, 
                'repeat_interleave_real': True, 
                'freqs_dtype': freqs_dtype
            }
            
            dype_kwargs = {
                'dype': self.dype, 
                'current_timestep': self.current_timestep, 
                'dype_scale': self.dype_scale, 
                'dype_exponent': self.dype_exponent,
                'ntk_scale': scale_global,      
                'override_mscale': current_mscale 
            }

            # Apply YaRN to spatial axes (1, 2)
            if i > 0:
                scale_local = max(1.0, current_patches / self.base_patches)
                dype_kwargs['linear_scale'] = scale_local 
                
                if scale_global > 1.0:
                    cos, sin = get_1d_dype_yarn_pos_embed(
                        **common_kwargs,
                        ori_max_pe_len=self.base_patches,
                        **dype_kwargs
                    )
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            else:
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)
            
        return torch.cat(emb_parts, dim=-3)

    def _forward_yarn(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Standard YaRN
        """
        n_axes = pos.shape[-1]
        emb_parts = []
        
        current_patches_h = int(pos[..., 1].max().item() + 1)
        current_patches_w = int(pos[..., 2].max().item() + 1)
        max_current_patches = max(current_patches_h, current_patches_w)
        needs_extrapolation = (max_current_patches > self.base_patches)

        if needs_extrapolation and self.yarn_alt_scaling:
            # Anisotropic (High-Res)
            # works for some resolution
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                current_patches_on_axis = int(axis_pos.max().item() + 1)
                if i > 0 and current_patches_on_axis > self.base_patches:
                    max_pe_len = torch.tensor(current_patches_on_axis, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_yarn_pos_embed(**common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs, use_aggressive_mscale=True)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
                
                cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
                sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
                row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
                row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
                matrix = torch.stack([row1, row2], dim=-2)
                emb_parts.append(matrix)
            return torch.cat(emb_parts, dim=-3)

        else:
            # Isotropic (Stable)
            cos_full_spatial, sin_full_spatial = None, None
            if needs_extrapolation:
                spatial_axis_dim = self.axes_dim[1]
                square_pos = torch.arange(0, max_current_patches, device=pos.device).float()
                max_pe_len = torch.tensor(max_current_patches, dtype=freqs_dtype, device=pos.device)
                
                common_kwargs_spatial = {'dim': spatial_axis_dim, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                cos_full_spatial, sin_full_spatial = get_1d_yarn_pos_embed(
                    **common_kwargs_spatial, pos=square_pos, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs, use_aggressive_mscale=False
                )

            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                
                if i > 0 and needs_extrapolation:
                    pos_indices = axis_pos.long().view(-1)
                    cos = cos_full_spatial[pos_indices].view(axis_pos.shape[0], axis_pos.shape[1], -1)
                    sin = sin_full_spatial[pos_indices].view(axis_pos.shape[0], axis_pos.shape[1], -1)
                else:
                    common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

                cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
                sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
                row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
                row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
                matrix = torch.stack([row1, row2], dim=-2)
                emb_parts.append(matrix)
            
            return torch.cat(emb_parts, dim=-3)

    def _forward_ntk(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        n_axes = pos.shape[-1]
        emb_parts = []

        current_patches_h = int(pos[..., 1].max().item() + 1)
        current_patches_w = int(pos[..., 2].max().item() + 1)
        max_patches = max(current_patches_h, current_patches_w)
        unified_scale = max_patches / self.base_patches if max_patches > self.base_patches else 1.0

        # debug
        # if unified_scale > 1.0:
        #      print(f"[DyPE-NTK] Isotropic scale factor set to: {unified_scale:.4f}")

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            
            ntk_factor = 1.0
            if i > 0 and unified_scale > 1.0:
                base_ntk = unified_scale ** (axis_dim / (axis_dim - 2))
                if self.dype:
                    k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                    ntk_factor = base_ntk ** k_t
                else:
                    ntk_factor = base_ntk
                ntk_factor = max(1.0, ntk_factor)
            
            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)
            
        return torch.cat(emb_parts, dim=-3)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        if self.method == 'vision_yarn':
            emb = self._forward_vision_yarn(pos, freqs_dtype)
        elif self.method == 'yarn':
            emb = self._forward_yarn(pos, freqs_dtype)
        else: # 'ntk' or 'base'
            emb = self._forward_ntk(pos, freqs_dtype)
        
        return emb.unsqueeze(1).to(ids.device)

def apply_dype_to_flux(model: ModelPatcher, width: int, height: int, method: str, yarn_alt_scaling: bool, enable_dype: bool, dype_scale: float, dype_exponent: float, base_shift: float, max_shift: float) -> ModelPatcher:
    m = model.clone()
    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling)
    
    should_patch_schedule = True
    if hasattr(m.model, "_dype_params"):
        if m.model._dype_params == new_dype_params:
            should_patch_schedule = False
            # print("[DyPE] Parameters unchanged. Skipping patch.")
        else:
            pass
            # print("[DyPE] Parameters changed. Re-patching.")

    if enable_dype and should_patch_schedule:
        if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux):
            patch_size = m.model.diffusion_model.patch_size
            latent_h, latent_w = height // 8, width // 8
            padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
            image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)
            
            base_seq_len = 256
            max_seq_len = image_seq_len

            if max_seq_len <= base_seq_len:
                 dype_shift = base_shift
            else:
                slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                intercept = base_shift - slope * base_seq_len
                dype_shift = image_seq_len * slope + intercept
            
            dype_shift = max(0.0, dype_shift)
            # print(f"[DyPE DEBUG] dype_shift (mu): {dype_shift:.4f} for resolution {width}x{height}")

            class DypeModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST):
                pass

            new_model_sampler = DypeModelSamplingFlux(m.model.model_config)
            new_model_sampler.set_parameters(shift=dype_shift)
            
            m.add_object_patch("model_sampling", new_model_sampler)
            m.model._dype_params = new_dype_params 
    elif not enable_dype:
        if hasattr(m.model, "_dype_params"):
            class DefaultModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST): pass
            default_sampler = DefaultModelSamplingFlux(m.model.model_config)
            m.add_object_patch("model_sampling", default_sampler)
            del m.model._dype_params

    try:
        orig_embedder = m.model.diffusion_model.pe_embedder
        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX model.")

    new_pe_embedder = FluxPosEmbed(theta, axes_dim, method, yarn_alt_scaling, enable_dype, dype_scale, dype_exponent)
    m.add_object_patch("diffusion_model.pe_embedder", new_pe_embedder)
    
    sigma_max = m.model.model_sampling.sigma_max.item()
    def dype_wrapper_function(model_function, args_dict):
        timestep_tensor = args_dict.get("timestep")
        if timestep_tensor is not None and timestep_tensor.numel() > 0:
            current_sigma = timestep_tensor.item()
            if sigma_max > 0:
                normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                new_pe_embedder.set_timestep(normalized_timestep)
        
        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)
    
    return m