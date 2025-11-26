import torch
from comfy_api.latest import ComfyExtension, io
from .src.patch import apply_dype_to_flux

class DyPE_FLUX(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_FLUX",
            display_name="DyPE for FLUX",
            category="model_patches/unet",
            description="Applies DyPE (Dynamic Position Extrapolation) to a FLUX model for ultra-high-resolution generation.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The FLUX model to patch with DyPE.",
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image width. Must match the width of your empty latent."
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image height. Must match the height of your empty latent."
                ),
                io.Combo.Input(
                    "method",
                    options=["vision_yarn", "yarn", "ntk", "base"],
                    default="vision_yarn",
                    tooltip="Position encoding extrapolation method. 'vision_yarn' uses resonance-aware scaling (Best for anisotropic 4K).",
                ),
                io.Boolean.Input(
                    "yarn_alt_scaling",
                    default=False,
                    label_on="Anisotropic (High-Res)",
                    label_off="Isotropic (Stable Default)",
                    tooltip="[YARN Only] Alternate scaling for ultra-high resolutions.",
                ),
                io.Boolean.Input(
                    "enable_dype",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable Dynamic Position Extrapolation for RoPE.",
                ),
                io.Float.Input(
                    "dype_scale",
                    default=2.0, min=0.0, max=8.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE magnitude (λs)."
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0, min=0.0, max=8.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE progression over time (λt)."
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.5, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Base shift for the noise schedule (mu)."
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Max shift for the noise schedule (mu) at high resolutions."
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The FLUX model patched with DyPE.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, method: str, yarn_alt_scaling: bool, enable_dype: bool, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_shift: float = 0.5, max_shift: float = 1.15) -> io.NodeOutput:
        if not hasattr(model.model, "diffusion_model") or not hasattr(model.model.diffusion_model, "pe_embedder"):
             raise ValueError("This node is only compatible with FLUX models.")
        
        patched_model = apply_dype_to_flux(model, width, height, method, yarn_alt_scaling, enable_dype, dype_scale, dype_exponent, base_shift, max_shift)
        return io.NodeOutput(patched_model)

class DyPEExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_FLUX]

async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()