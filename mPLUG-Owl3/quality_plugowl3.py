from copy import deepcopy
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
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer



class QualityOwl3Model(nn.Module):
    def __init__(
        self,
        new_layers=None,
        prompt_len=0,  # Prompt长度
        model_path = "iic/mPLUG-Owl3-7B-241101",
        # 添加 LoRA 相关参数
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        trainable_prompt=True
    ):
        super().__init__()

        self.new_layers = new_layers
        self.quality_encoder = quality_backbone()
        self.grad_flag = False
        # 保存 LoRA 参数
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.trainable_prompt = trainable_prompt

        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        if new_layers is not None:
            config.hyper_layers += new_layers

        self.LLM = transformers.AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        LLM_lora = deepcopy(self.LLM)
        self.LLM.requires_grad_(False)
        
        # 为new_layers之后的层添加LoRA
        if hasattr(self, 'lora_r') and self.lora_r > 0:
            # 确定要添加LoRA的层索引
            num_layers = len(self.LLM.language_model.model.layers)
            target_layer_idx = [i for i in range(self.new_layers[-1]-1, num_layers)] #max(self.new_layers) if self.new_layers else 0
        
            # 创建LoRA配置
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "k_proj", "o_proj"],
            )

            # 对target_layer_idx应用LoRA
            get_peft_model(LLM_lora.language_model.model, peft_config)
            for l in target_layer_idx:
                self.LLM.language_model.model.layers[l] = LLM_lora.language_model.model.layers[l]


        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.processor = self.LLM.init_processor(self.tokenizer)

        self.quality2text_model = nn.Sequential(
            nn.Linear(768, self.LLM.embed_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.LLM.embed_dim, self.LLM.embed_dim),
        )


        # Prompt-tuning specific
        self.prompt_len = prompt_len
        self.embed_dim = self.LLM.embed_dim
        self.prompt_embeddings = nn.Embedding(prompt_len, self.embed_dim)
        # ------------------------------
        # 1. 先随机初始化
        nn.init.xavier_normal_(self.prompt_embeddings.weight)

        # 2. 获取目标字符串的Token embedding
        init_prompt_str = "Analize from details, how would you rate the quality of this image?"
        with torch.no_grad():
            encoded = self.tokenizer(
                init_prompt_str, return_tensors="pt"
            )
            # 获取已冻结的embed_tokens
            question_embeds = self.LLM.language_model.model.embed_tokens(
                encoded["input_ids"]
            )[0]  # shape: [seq_len, embed_dim]

        # 3. 根据prompt_len截断或填补
        seq_len = question_embeds.size(0)
        if seq_len >= prompt_len:
            question_embeds = question_embeds[:prompt_len, :]
        else:
            # 剩余位置可维持原来的随机初始化，也可以padding为0
            pass

        # 4. 覆写prompt_embeddings的前seq_len行
        self.prompt_embeddings.weight.data[:seq_len] = question_embeds.data


        self.load_frozen_state_dict()

    def load_frozen_state_dict(self):
        # ...原有代码保持不变...
        self.quality_encoder.load_state_dict(
            torch.load("exps/weights/fragments_model.pth", map_location="cpu"),
            strict=True,
        )
        self.logi_indices = torch.load("exps/lsvq/indices_lsvq.pt", map_location="cpu")
        sentiment_weight = torch.load(
            "exps/lsvq/linear300_0.7911.pth", map_location="cpu"
        )
        self.linear_sw = nn.Linear(300, 1)
        self.linear_sw.load_state_dict(sentiment_weight)
    
        # 冻结所有参数
        self.quality_encoder.requires_grad_(False)
        
        # 为v_kv_proj单独设置梯度
        for l in self.new_layers:
            self.LLM.language_model.model.layers[
                l - 1
            ].self_attn.v_kv_proj.requires_grad_(True)
        

        # 其他参数设置
        self.quality2text_model.requires_grad_(True)
        # 冻结LLM的embedding层
        self.LLM.language_model.model.embed_tokens.requires_grad_(False)
        # 只训练prompt embedding
        self.prompt_embeddings.requires_grad_(self.trainable_prompt)


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

        # Prepend the prompt embeddings to the input embeddings
        batch_size = inputs_embeds.shape[0]
        prompt_embeds = self.prompt_embeddings.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        ).to(dtype=inputs_embeds.dtype)  # [batch_size, prompt_len, embed_dim]
        inputs_embeds = torch.cat(
            (prompt_embeds, inputs_embeds), dim=1
        )  # [batch_size, prompt_len + seq_length, embed_dim]

        # Update attention mask and position IDs
        if attention_mask is None:
            # 如果attention_mask为None，创建一个全1的掩码
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        # 现在安全地连接prompt部分的掩码
        attention_mask = torch.cat(
            (
                torch.ones(batch_size, self.prompt_len, device=inputs_embeds.device),
                attention_mask,
            ),
            dim=1,
        )

        # 在拼接prompt embeddings之前检查position_ids的batch维度
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            # 确保position_ids有batch_size维度
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            # 如果提供的position_ids只有一个batch但需要多个
            if position_ids.size(0) == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            position_ids = position_ids.view(batch_size, -1).long()

        position_ids = torch.cat(
            (
                torch.arange(0, self.prompt_len, device=inputs_embeds.device).unsqueeze(
                    0
                ).expand(batch_size, -1),
                position_ids,
            ),
            dim=1,
        )

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
            if decoder_layer.layer_idx + 1 in self.new_layers:
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

        # 处理每个batch的消息 - 简化消息内容，因为问题已在prompt_embeddings中
        all_inputs = []
        for i in range(batch_size):
            msg = [
                {
                    "role": "user", 
                    "content": "<|video|>",  # 只保留视频标记，问题由prompt_embeddings提供
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

    
        # 处理视觉信息，不需要梯度
        with torch.no_grad():
            quality_features = self.quality_encoder(technical)  # [B, C, D, H, W]
            image_embeds = self.LLM.forward_image(batched_inputs.pop("pixel_values"))

        qf = rearrange(quality_features, "b c d h w -> (b d) (h w) c")
        quality_embed = self.quality2text_model(qf).bfloat16()
    
        
        outputs = self.HyperQwen2ForCausalLM_forward(
            image_embeds=image_embeds,
            quality_embed=quality_embed, 
            labels=labels,
            **batched_inputs,
        )

        # 显存不足，删除一些不需要的变量  #和模型参数
        del quality_features, qf, image_embeds, aesthetic, technical, labels
        torch.cuda.empty_cache()
    
        return outputs
    
    @property
    def dev(self):
        return self.LLM.device






if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QualityOwl3Model().to(dev)
    # logits=[]
    # names=[]
    # with torch.no_grad():
    #     for data in tqdm.tqdm(dl):
    #         o = model.forward(**data)
    #         names += data["name"]
    #         logits.append(o.logits[:,-1].cpu())
    # logits_all = torch.stack(logits)
    # lmax = torch.max(logits_all, dim=0)
    # top300max = torch.topk(lmax.values, 300, dim=-1)
    # logits_top300 = logits_all[:,top300max.indices]
    # logits_dict = {}
    # for i, img in enumerate(names):
    #     logits_dict[img] = logits_top300[i]
    # torch.save(logits_dict, "exps/lsvq/logits_lsvq.pt")
    # torch.save(top300max.indices, "exps/lsvq/indices_lsvq.pt")
