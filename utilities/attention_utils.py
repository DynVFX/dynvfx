from utilities.utils import isinstance_str
import torch
import torch.nn.functional as F
from einops import rearrange
import torch.backends.cuda


def get_text_embeds(pipeline, target_prompt, negative_prompt, device="cuda"):
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=target_prompt,
        device=device,
        num_videos_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    text_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

    return text_embeds


def register_module_property(pipeline, property_name, property_value, module_type=None):
    """
    Generic function to register properties on modules.

    Args:
        pipeline: The diffusion pipeline
        property_name: Name of the property to set
        property_value: Value to set for the property
        module_type: Type of modules to set property on (default: ["ModuleWithExtendedAttention"])
    """
    if module_type is None:
        module_type = ["ModuleWithExtendedAttention"]

    for _, module in pipeline.transformer.named_modules():
        if isinstance_str(module, module_type):
            setattr(module, property_name, property_value)


class ModuleWithExtendedAttention(torch.nn.Module):
    # Constants
    LATENT_DOWNSCALE_FACTOR = 8 # compression factor from VAE
    FEATURE_DOWNSCALE_FACTOR = 2 # compression factor in transformer
    LATENT_NUM_FRAMES = 13 # Fixed number of latent frames for CogVideoX
    TEXT_SEQ_LENGTH = 226 # Text sequence length for CogVideoX


    def __init__(
            self,
            module,
            height,
            width,
            block_name,
            precision,
            mask_injection,
            selective_attention_type,
            extend_st_for_text_keys,
            fg_dropout_percentage,
            bg_dropout_percentage,
    ):
        super().__init__()
        self.module = module
        self.h = height
        self.w = width
        self.precision = precision
        self.feature_h = self.h // self.LATENT_DOWNSCALE_FACTOR // self.FEATURE_DOWNSCALE_FACTOR
        self.feature_w = self.w // self.LATENT_DOWNSCALE_FACTOR // self.FEATURE_DOWNSCALE_FACTOR
        self.latent_num_frames = self.LATENT_NUM_FRAMES
        self.block_name = block_name
        self.text_seq_length = self.TEXT_SEQ_LENGTH
        self.selective_attention_type = selective_attention_type
        self.extend_st_for_text_keys = extend_st_for_text_keys
        self.masks_for_attn = None
        self.with_injection = True
        self.mask_injection = mask_injection
        self.fg_dropout_percentage = fg_dropout_percentage
        self.bg_dropout_percentage = bg_dropout_percentage

    def apply_selective_attention(self, query, key_cond, value_cond, source_key, source_value, text_seq_length, attention_mask):
        """
        Apply selective attention using extended attention mechanisms.

        This implements the core algorithm from the paper, supporting three modes:
        1. Full Extended Attention: K_cond ∪ K_orig^E
        2. Masked Extended Attention: K_cond ∪ F(K_orig^E) where F uses M_orig
        3. Anchor Extended Attention: K_cond ∪ F(K_orig^E) where F uses dropout

        Args:
            query: Query tensor from target video
            key_cond: Conditional key tensor
            value_cond: Conditional value tensor
            source_key: Source key tensor (K_orig in the paper)
            source_value: Source value tensor (V_orig in the paper)
            text_seq_length: Length of text sequence
            attention_mask: Optional attention mask

        Returns:
            Modified hidden states with injected selective attention
        """
        assert self.with_injection, "injection must be enabled"

        # Extract spatial components
        st_source_key = source_key[:, :, text_seq_length:, :].clone()
        st_source_value = source_value[:, :, text_seq_length:, :].clone()

        # Determine which type of selective attention to apply based on configuration
        assert self.selective_attention_type in ["full", "masked", "anchor"]

        if self.selective_attention_type == "full":
            st_key_cond = torch.cat([key_cond, st_source_key], dim=-2)
            st_value_cond = torch.cat([value_cond, st_source_value], dim=-2)

        else:
            # Masked or Anchor Extended Attention - requires masks
            assert self.masks_for_attn is not None
            st_key_cond = key_cond
            st_value_cond = value_cond
            st_source_key_rerange = rearrange(
                st_source_key,
                "1 head (f h w) d -> 1 head f h w d",
                f=self.latent_num_frames,
                h=self.feature_h,
                w=self.feature_w,
            )
            st_source_value_rerange = rearrange(
                st_source_value,
                "1 head (f h w) d -> 1 head f h w d",
                f=self.latent_num_frames,
                h=self.feature_h,
                w=self.feature_w,
            )
            latent_morig = self.masks_for_attn
            latent_morig = rearrange(
                latent_morig,
                "f c h w -> c f h w"
            )

            if self.selective_attention_type == "anchor":
                # For Anchor Extended Attention, apply the selection function F from Eq. (3)
                # F := { DropFG(Morig) ∪ DropBG(~Morig) } ∘ A

                # Generate foreground dropout pattern (DropFG)
                fg_random_pattern = torch.bernoulli(
                    torch.ones(1, 1, self.feature_h, self.feature_w,
                               dtype=self.precision, device=latent_morig.device) *
                    self.fg_dropout_percentage)
                # Generate background dropout pattern (DropBG)
                bg_random_pattern = torch.bernoulli(
                    torch.ones(1, 1, self.feature_h, self.feature_w,
                               dtype=self.precision, device=latent_morig.device) *
                    self.bg_dropout_percentage
                )

                # Create the anchor positions (F(Morig))
                anchor_positions = torch.where(
                    latent_morig > 0,
                    latent_morig * fg_random_pattern,  #  DropFG(Morig)
                    bg_random_pattern  # DropBG(~Morig)
                )
            else:
                # For Masked Extended Attention, use the original mask directly
                # F(A) = Morig ∘ A
                anchor_positions = latent_morig

            for i in range(self.latent_num_frames):
                # Get indices where anchor positions are non-zero
                nonzero_indices = anchor_positions[0][i].nonzero()
                h_indices = nonzero_indices[:, 0]
                w_indices = nonzero_indices[:, 1]

                # Extract features at anchor positions (K^E and V^E in the paper)
                st_key_extend = st_source_key_rerange[0, :, i, h_indices, w_indices, :].clone().unsqueeze(0)
                st_value_extend = st_source_value_rerange[0, :, i, h_indices, w_indices, :].clone().unsqueeze(0)

                st_key_cond = torch.cat([st_key_cond, st_key_extend], dim=-2)
                st_value_cond = torch.cat([st_value_cond, st_value_extend], dim=-2)

        return self.apply_extended_attention(query, key_cond, value_cond, st_key_cond, st_value_cond, text_seq_length, attention_mask)

    def apply_extended_attention(self, query, key_cond, value_cond, st_key_cond, st_value_cond, text_seq_length, attention_mask):
        """
        Apply selective attention to text and spatial components.

        Args:
            query: Query tensor
            key_cond: Original conditional key tensor
            value_cond: Original conditional value tensor
            st_key_cond: Spatial conditional key tensor
            st_value_cond: Spatial conditional value tensor
            text_seq_length: Length of text sequence
            attention_mask: Attention mask

        Returns:
            torch.Tensor: Combined hidden states
        """
        # Extract text query
        st_query_cond = query[[0]][:, :, text_seq_length:, :]
        text_query_cond = query[[0]][:, :, :text_seq_length, :]

        # Ensure proper conditioning for text keys
        st_key_cond[:, :, :text_seq_length, :] = st_key_cond[:, :, :text_seq_length, :]

        # Apply attention based on configuration
        if self.extend_st_for_text_keys:
            text_hidden_states_cond = F.scaled_dot_product_attention(
                text_query_cond, st_key_cond, st_value_cond, attn_mask=attention_mask, dropout_p=0.0,
                is_causal=False
            )
        else:
            text_hidden_states_cond = F.scaled_dot_product_attention(
                text_query_cond, key_cond, value_cond, attn_mask=attention_mask, dropout_p=0.0,
                is_causal=False
            )

        # Apply attention to spatial tokens
        st_hidden_states_cond = F.scaled_dot_product_attention(
            st_query_cond, st_key_cond, st_value_cond, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False
        )

        # Combine results
        hidden_states_cond = torch.cat([text_hidden_states_cond, st_hidden_states_cond], dim=-2)

        return hidden_states_cond

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, image_rotary_emb=None):
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.module.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.module.heads, -1, attention_mask.shape[-1])

        query = self.module.to_q(hidden_states)
        key = self.module.to_k(hidden_states)
        value = self.module.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.module.heads

        query = query.view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)

        if self.module.norm_q is not None:
            query = self.module.norm_q(query)
        if self.module.norm_k is not None:
            key = self.module.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not self.module.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        if self.with_injection and (query.shape[0] > 1):
            with torch.backends.cuda.sdp_kernel(enable_math=True):
                key_cond, key_uncond = key.chunk(2)
                source_key = key_uncond.clone()
                value_cond, value_uncond = value.chunk(2)
                source_value = value_uncond.clone()
                hidden_states_cond = self.apply_selective_attention(query, key_cond, value_cond, source_key, source_value, text_seq_length, attention_mask)
                hidden_states_uncond = F.scaled_dot_product_attention(
                    query[[1]], key[[1]], value[[1]], attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = torch.cat([hidden_states_cond, hidden_states_uncond], dim=0)
        else:
            with torch.backends.cuda.sdp_kernel(enable_math=True):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.module.heads * head_dim) # linear proj
        hidden_states = self.module.to_out[0](hidden_states)  # dropout
        hidden_states = self.module.to_out[1](hidden_states)
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


def register_attention_guidance(model):
    """
    Register attention guidance by replacing attention modules with extended attention modules.

    Args:
        model: The model to modify
    """
    # Extract model parameters
    height = model.input_tensor.shape[2]
    width = model.input_tensor.shape[3]
    precision = model.precision

    # Extract configuration parameters
    mask_injection = model.config["mask_injection"]
    selective_attention_type = model.config["selective_attention_type"]
    extend_st_for_text_keys = model.config["extend_st_for_text_keys"]
    fg_dropout_percentage = model.config["foreground_dropout_percentage"]
    bg_dropout_percentage = model.config["background_dropout_percentage"]

    # Only apply the modification if injection is enabled
    if model.config["with_injection"]:
        inject_attn_transformer_layers = model.config["inject_attn_transformer_layers"]

        # Iterate through transformer blocks and replace attention modules
        for layer in range(len(model.pipeline.transformer.transformer_blocks)):
            transformer_block = model.pipeline.transformer.transformer_blocks[layer]
            block_name = f"transformer_block{layer}"

            # Apply only to specified layers in inject_attn_transformer_layers
            if layer in inject_attn_transformer_layers:
                transformer_block.attn1 = ModuleWithExtendedAttention(
                    transformer_block.attn1,
                    height,
                    width,
                    block_name=block_name,
                    precision=precision,
                    mask_injection=mask_injection,
                    selective_attention_type=selective_attention_type,
                    extend_st_for_text_keys=extend_st_for_text_keys,
                    fg_dropout_percentage=fg_dropout_percentage,
                    bg_dropout_percentage=bg_dropout_percentage,
                )