import torch, sys, transformers
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from swin_backbone import swin_3d_tiny as quality_backbone
from swin_backbone import get_spatial_fragments
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
from einops import rearrange, repeat


MAX_NUM_FRAMES = 32
model_path = "iic/mPLUG-Owl3-7B-241101"
dev = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def encode_video(video_path, num_frames=32):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    
    sample_fps = max(len(vr) // num_frames, 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    print("num frames:", len(frames))
    return frames


class QualityOwl3Model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.quality_encoder = quality_backbone()
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        config.hyper_layers.append(27)
        self.LLM = transformers.AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            attn_implementation='flash_attention_2', 
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.processor = self.LLM.init_processor(self.tokenizer)

        self.quality2text_model = nn.Sequential(
            nn.Linear(768, self.LLM.embed_dim),
            nn.GELU(),
            nn.Linear(self.LLM.embed_dim, self.LLM.embed_dim),
        )



        self.msg = [
            {
                "role": "user",
                "content": """"<|video|>"Analize from details, how would you rate the quality of this image?""",
            },
            {"role": "assistant", "content": "The quality of the image is very"},
        ]

    def forward(
        self,
        videos=None,
        qvideos=None,
        labels=None,
    ):

        inputs = self.processor(self.msg, images=None, videos=videos, preface=True).to(dev)

        # 处理质量评估视频特征
        quality_features = self.quality_encoder(qvideos.unsqueeze(0))
        qf = rearrange(quality_features, "n c d h w -> (n d) (h w) c")
        quality_embed = self.quality2text_model(qf).bfloat16()

        image_embeds = self.LLM.forward_image(inputs.pop("pixel_values"))
        # (24, 729, self.LLM.embed_dim:=3584)

        # outputs = self.LLM.language_model(image_embeds=image_embeds, **inputs,)
        ########## self.LLM.language_model ##### class HyperQwen2ForCausalLM ###########
        output_attentions = inputs.pop("output_attentions", None)
        output_hidden_states = inputs.pop("output_hidden_states", None)
        return_dict = inputs.pop("return_dict", None)
        output_attentions = output_attentions if output_attentions is not None else self.LLM.language_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.LLM.language_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.LLM.language_model.config.use_return_dict

        
        # outputs = self.LLM.language_model.model(
        #     image_embeds=image_embeds, 
        #     **inputs,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     )
        ########## self.LLM.language_model.model ##### class HyperQwen2Model ###########
        from transformers.cache_utils import Cache, DynamicCache
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
        from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
        input_ids = inputs.pop("input_ids", None)
        inputs_embeds = inputs.pop("inputs_embeds", None)
        attention_mask = inputs.pop("attention_mask", None)
        position_ids = inputs.pop("position_ids", None)
        past_key_values = inputs.pop("past_key_values", None)
        media_offset = inputs.pop("media_offset", None)
        use_cache = inputs.pop("use_cache", None)
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.LLM.language_model.model.gradient_checkpointing and self.LLM.language_model.model.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.LLM.language_model.model.embed_tokens(input_ids)

        if attention_mask is not None and self.LLM.language_model.model._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self.LLM.language_model.model._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.LLM.language_model.model._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.LLM.language_model.model.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.LLM.language_model.model.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # beam search
        if batch_size != len(media_offset):
            # The model is performing beamsearch, repeat the visual content
            beam_factor = batch_size // len(media_offset)
            assert batch_size % len(media_offset) == 0
            media_offset = media_offset * beam_factor
            image_embeds = repeat(image_embeds, 'B L D -> (factor B) L D', factor=beam_factor)

        # # Flex mask
        
        # expected_q_len =  hidden_states.shape[1]
        # expected_kv_len = expected_q_len
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, 1)
        
        #     length_each_img = image_embeds.shape[1]
        #     expected_kv_len = [expected_kv_len+len(_)*length_each_img for _ in media_offset]
        #     flex_mask_block = []

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.LLM.language_model.model.layers:
            if decoder_layer.layer_idx == 26:
                image_embeds = quality_embed
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.LLM.language_model.model.gradient_checkpointing and self.LLM.language_model.model.training:
                layer_outputs = self.LLM.language_model.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    image_embeds,
                    media_offset,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_embeds=image_embeds,
                    media_offset=media_offset,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.LLM.language_model.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        #####over##### self.LLM.language_model.model ##### class HyperQwen2Model ###########

        
        hidden_states = outputs[0]
        logits = self.LLM.language_model.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        ####### over ### self.LLM.language_model ##### class HyperQwen2ForCausalLM ###########


        return self.tokenizer.decode(torch.topk(outputs.logits[0,-1], k=50).indices)


if __name__ == "__main__":
    vids = ["/home/ippl/xxr/datasets/MaxWell/0049.mp4"]
    img_lists = [encode_video(vid) for vid in vids]
    video = torch.stack(
        [transforms.ToTensor()(frame) for frame in img_lists[0]]
    )  # [T, C, H, W]
    video = video.permute(1, 0, 2, 3)  # Add batch dimension: [C, T, H, W]
    qvideo = get_spatial_fragments(video, aligned=8).to(dev)
    v = {"videos": [img_list[::2] for img_list in img_lists], "qvideos": qvideo}

    m = QualityOwl3Model().to(dev)
    m.eval()
    with torch.no_grad():
        outputs = m(**v)
    print(outputs)
