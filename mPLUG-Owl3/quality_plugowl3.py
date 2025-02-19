import torch, sys, transformers
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from swin_backbone import SwinTransformer3D as quality_backbone

from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
from einops import rearrange, repeat

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

model_path = "iic/mPLUG-Owl3-7B-241101"



class QualityOwl3Model(nn.Module):
    def __init__(
        self,
        tech_brance=True
    ):
        super().__init__()

        self.quality_encoder = quality_backbone()
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        if tech_brance:
            config.hyper_layers.append(27)

        self.LLM = transformers.AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.processor = self.LLM.init_processor(self.tokenizer)

        self.quality2text_model = nn.Sequential(
            nn.Linear(768, self.LLM.embed_dim),
            nn.GELU(),
            nn.Linear(self.LLM.embed_dim, self.LLM.embed_dim),
        )

        self.load_frozen_state_dict()


    def load_frozen_state_dict(self):
        self.quality_encoder.load_state_dict(torch.load("exps/weights/fragments_model.pth"),strict=True)
        self.logi_indices = torch.load("exps/lsvq/indices_lsvq.pt")
        sentiment_weight = torch.load("exps/lsvq/linear300_0.7911.pth")
        self.linear_sw = nn.Linear(300, 1)
        self.linear_sw.load_state_dict(sentiment_weight)

        self.quality_encoder.requires_grad_(False)
        self.LLM.requires_grad_(False)
        self.LLM.language_model.model.layers[26].self_attn.v_kv_proj.requires_grad_(True)
        self.quality2text_model.requires_grad_(True)



    def HyperQwen2Model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds=None,
        media_offset=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        quality_embed=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if (
            self.LLM.language_model.model.gradient_checkpointing
            and self.LLM.language_model.model.training
        ):
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
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.LLM.language_model.model.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and self.LLM.language_model.model._attn_implementation
            == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self.LLM.language_model.model._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif (
            self.LLM.language_model.model._attn_implementation == "sdpa"
            and not output_attentions
        ):
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
            image_embeds = repeat(
                image_embeds, "B L D -> (factor B) L D", factor=beam_factor
            )

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
            if (
                self.LLM.language_model.model.gradient_checkpointing
                and self.LLM.language_model.model.training
            ):
                layer_outputs = (
                    self.LLM.language_model.model._gradient_checkpointing_func(
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
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        return outputs

    def HyperQwen2ForCausalLM_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds=None,
        media_offset=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        quality_embed=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.LLM.language_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.LLM.language_model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.LLM.language_model.config.use_return_dict
        )

        # outputs = self.LLM.language_model.model(
        #     image_embeds=image_embeds,
        #     **inputs,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     )
        ########## self.LLM.language_model.model ##### class HyperQwen2Model ###########
        outputs = self.HyperQwen2Model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            media_offset=media_offset,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            quality_embed=quality_embed,
        )
        #####over##### self.LLM.language_model.model ##### class HyperQwen2Model ###########

        hidden_states = outputs[0]
        logits = self.LLM.language_model.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        score = None
        if labels is not None:
            last_logits = logits[..., -1, :].contiguous()
            top_logits = last_logits[:,self.logi_indices]
            score = self.linear_sw(top_logits).squeeze(-1)
            plccloss = self.plcc_loss(score, labels)
            rankloss = self.rank_loss(score, labels)
            loss = (plccloss, rankloss, score)

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

        return outputs

    def forward_OLD(
        self,
        aesthetic=None,
        technical=None,
        labels=None,
        **args,
    ):
        if not isinstance(aesthetic[0], list):
            aesthetic = [aesthetic]
        msg = [
            {
                "role": "user",
                "content": """"<|video|>"Analize from details, how would you rate the quality of this image?""",
            },
            {"role": "assistant", "content": "The quality of the image is very"},
        ]
        inputs = self.processor(msg, images=None, videos=aesthetic, preface=True).to(self.dev)

        # 处理质量评估视频特征
        quality_features = self.quality_encoder(technical.unsqueeze(0).to(self.dev))
        qf = rearrange(quality_features, "n c d h w -> (n d) (h w) c")
        quality_embed = self.quality2text_model(qf).bfloat16()

        image_embeds = self.LLM.forward_image(inputs.pop("pixel_values"))
        # (24, 729, self.LLM.embed_dim:=3584)

        # outputs = self.LLM.language_model(image_embeds=image_embeds, **inputs,)
        ########## self.LLM.language_model ##### class HyperQwen2ForCausalLM ###########
        outputs = self.HyperQwen2ForCausalLM_forward(
            image_embeds=image_embeds,
            quality_embed=quality_embed,
            labels=labels,
            **inputs,
        )
        ####### over ### self.LLM.language_model ##### class HyperQwen2ForCausalLM ###########

        return outputs.logits


    def rank_loss(self, y_pred, y):
        ranking_loss = torch.nn.functional.relu(
            (y_pred - y_pred.t()) * torch.sign((y.t() - y))
        )
        scale = 1 + torch.max(ranking_loss)
        return (
            torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
        ).float()

    def plcc_loss(self, y_pred, y):
        sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
        y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
        sigma, m = torch.std_mean(y, unbiased=False)
        y = (y - m) / (sigma + 1e-8)
        loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
        rho = torch.mean(y_pred * y)
        loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
        return ((loss0 + loss1) / 2).float()# + 0.3 * rank_loss(y_pred[...,None], y[...,None])

    def forward(self, aesthetic=None, technical=None, labels=None, **args):
        batch_size = len(aesthetic)
        
        # 处理technical数据转为tensor
        if isinstance(technical, list):
            technical = torch.stack(technical)
        technical = technical.to(self.dev)

        # 处理每个batch的消息
        all_inputs = []
        for i in range(batch_size):
            msg = [
                {
                    "role": "user", 
                    "content": """"<|video|>"Analize from details, how would you rate the quality of this image?""",
                },
                {"role": "assistant", "content": "The quality of the image is very"},
            ]
            # 分别处理每个样本
            inputs = self.processor(msg, images=None, videos=[aesthetic[i]], preface=True).to(self.dev)
            all_inputs.append(inputs)
        
        # 修改batched_inputs的处理方式
        batched_inputs = {}
        for key in all_inputs[0].keys():
            if isinstance(all_inputs[0][key], torch.Tensor):
                batched_inputs[key] = torch.cat([inp[key] for inp in all_inputs], dim=0)
            elif key == "media_offset":
                # 特殊处理media_offset
                media_tensors = [inp["media_offset"][0] for inp in all_inputs]  # 取出每个列表中的tensor
                batched_inputs["media_offset"] = torch.stack(media_tensors, dim=0)  # [B, seq_len]

    
        # 处理technical特征，支持批处理
        quality_features = self.quality_encoder(technical)  # [B, C, D, H, W]
        qf = rearrange(quality_features, "b c d h w -> (b d) (h w) c")
        quality_embed = self.quality2text_model(qf).bfloat16()
    
        image_embeds = self.LLM.forward_image(batched_inputs.pop("pixel_values"))
        
        outputs = self.HyperQwen2ForCausalLM_forward(
            image_embeds=image_embeds,
            quality_embed=quality_embed, 
            labels=labels,
            **batched_inputs,
        )
    
        return outputs
    
    @property
    def dev(self):
        return self.LLM.device

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import yaml
    import tqdm
    from exps.fitting import Owl3logits, fit_linear, list_path
    from dataset import ViewDecompositionDataset, dict_simply_collate

    opt = yaml.safe_load(open("data.yml"))
    d = ViewDecompositionDataset(opt["train"])
    dl = torch.utils.data.DataLoader(d, batch_size=2, shuffle=False, num_workers=16, collate_fn=dict_simply_collate)
    model = QualityOwl3Model(tech_brance=False).to(dev)
    logits=[]
    names=[]
    with torch.no_grad():
        for data in tqdm.tqdm(dl):
            o = model.forward(**data)
            names += data["name"]
            logits.append(o.logits[:,-1].cpu())
    logits_all = torch.stack(logits)
    lmax = torch.max(logits_all, dim=0)
    top300max = torch.topk(lmax.values, 300, dim=-1)
    logits_top300 = logits_all[:,top300max.indices]
    logits_dict = {}
    for i, img in enumerate(names):
        logits_dict[img] = logits_top300[i]
    torch.save(logits_dict, "exps/lsvq/logits_lsvq.pt")
    torch.save(top300max.indices, "exps/lsvq/indices_lsvq.pt")
