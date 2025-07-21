# use env: /root/anaconda3/bin/python
from copy import deepcopy
import pandas as pd
import tqdm
import torch, sys, transformers, os
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import dataset
from swin_backbone import SwinTransformer3D as quality_backbone
from dover import VQABackbone, VQAHead
import ffmpeg
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
from exps import fitting
from einops import rearrange, repeat
import random

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
import os,json
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1" # 去除颜色编码
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["AV_LOG_LEVEL"] = "quiet"  
os.environ["FFMPEG_LOGLEVEL"] = "quiet" 

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import dataset
from quality_plugowl3 import QualityOwl3Model
import yaml
import tqdm, time
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

    
def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()# + 0.3 * rank_loss(y_pred[...,None], y[...,None])

class QualityOwl3Model(nn.Module):
    def __init__(
        self,
        new_layers=None,
        prompt_len=0,  # Prompt长度
        model_path = "iic/mPLUG-Owl3-7B-241101",
        # 添加 LoRA 相关参数
        lora_r=8,
        lora_alpha=None,
        lora_dropout=0.05,
        trainable_prompt=True
    ):
        super().__init__()

        self.new_layers = new_layers

        # 保存 LoRA 参数
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha if lora_alpha is not None else 2 * lora_r
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
            local_files_only=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.LLMprocessor = self.LLM.init_processor(self.tokenizer)

        self.load_frozen_state_dict()
        if hasattr(self, 'lora_r') and self.lora_r > 0:
            self.set_lora()

    def load_frozen_state_dict(self):

        self.logi_indices = torch.load("exps/lsvq/indices_lsvq.pt", map_location="cpu")
        sentiment_weight = torch.load(
            "exps/lsvq/linear300_0.7911.pth", map_location="cpu"
        )
        self.linear_sw = nn.Linear(300, 1)
        self.linear_sw.load_state_dict(sentiment_weight)
    
    def set_lora(self):
        LLM_lora = deepcopy(self.LLM)
        self.LLM.requires_grad_(False)
        
        # 为new_layers之后的层添加LoRA
        
        # 确定要添加LoRA的层索引
        num_layers = len(self.LLM.language_model.model.layers)
        target_layer_idx = [i for i in range(26, num_layers)] # 从最后一个image_embeds交互时开始lora微调
        # target_layer_idx = [i for i in range(self.new_layers[-1]-1, num_layers)] # 只微调quality_embeds交互时的lora
        # target_layer_idx = [26] # 只微调最后一层的lora
    
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

        # # Prepend the prompt embeddings to the input embeddings
        # batch_size = inputs_embeds.shape[0]
        # prompt_embeds = self.prompt_embeddings.weight.unsqueeze(0).expand(
        #     batch_size, -1, -1
        # ).to(dtype=inputs_embeds.dtype)  # [batch_size, prompt_len, embed_dim]
        # inputs_embeds = torch.cat(
        #     (prompt_embeds, inputs_embeds), dim=1
        # )  # [batch_size, prompt_len + seq_length, embed_dim]

        # # Update attention mask and position IDs
        # if attention_mask is None:
        #     # 如果attention_mask为None，创建一个全1的掩码
        #     attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        # # 现在安全地连接prompt部分的掩码
        # attention_mask = torch.cat(
        #     (
        #         torch.ones(batch_size, self.prompt_len, device=inputs_embeds.device),
        #         attention_mask,
        #     ),
        #     dim=1,
        # )

        # # 在拼接prompt embeddings之前检查position_ids的batch维度
        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     position_ids = torch.arange(
        #         past_key_values_length,
        #         seq_length + past_key_values_length,
        #         dtype=torch.long,
        #         device=device,
        #     )
        #     # 确保position_ids有batch_size维度
        #     position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # else:
        #     # 如果提供的position_ids只有一个batch但需要多个
        #     if position_ids.size(0) == 1 and batch_size > 1:
        #         position_ids = position_ids.expand(batch_size, -1)
        #     position_ids = position_ids.view(batch_size, -1).long()

        # position_ids = torch.cat(
        #     (
        #         torch.arange(0, self.prompt_len, device=inputs_embeds.device).unsqueeze(
        #             0
        #         ).expand(batch_size, -1),
        #         position_ids,
        #     ),
        #     dim=1,
        # )

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
            plccloss = plcc_loss(score, labels)
            rankloss = rank_loss(score, labels)
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
    

    def processor(self, data_and_info):
        if not isinstance(data_and_info, list) and isinstance(data_and_info, dict):
            image_or_video = data_and_info["data"]
            media_info = data_and_info["info"]
        elif isinstance(data_and_info, list) and isinstance(data_and_info[0], dict):
            image_or_video = [d["data"] for d in data_and_info]
            media_info = [d["info"] for d in data_and_info]
        media_type = "images" if isinstance(image_or_video[0], Image.Image) else "videos"
        batch_size = len(image_or_video)
        media_token = {"images": "<|image|>", "videos": "<|video|>"}

        # 准备批处理数据 (保持不变)
        batched_messages = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "user",
                    "content": f"{media_token[media_type]}The infomation of the {media_type[:-1]} is as follows:{media_info[i]}, how would you rate the quality of this {media_type[:-1]}?",
                },
                {"role": "assistant", "content": "The quality of the image is very"},
            ]
            batched_messages.append(messages)

        # --- 开始手动处理和填充 ---
        processed_outputs = []
        all_pixel_values = [] # 用于收集 pixel_values

        # 1. 逐个处理样本
        for i in range(batch_size):
            current_messages = batched_messages[i]
            current_media = [image_or_video[i]] # LLMprocessor 可能期望列表

            # 准备单个样本的输入字典
            single_process_dict = {
                "messages": current_messages,
                media_type: current_media, # 传递单个媒体项的列表
                "preface": True
            }

            # 调用 LLMprocessor 处理单个样本
            # 注意：这里不再需要 padding=True 或 return_tensors="pt" (虽然 return_tensors 仍可用于 pixel_values)
            try:
                # 假设 LLMprocessor 返回一个包含 'input_ids', 'media_offset', 'pixel_values' 的字典或对象
                # 确保返回的是 tensor，至少 pixel_values 需要是
                single_output = self.LLMprocessor(**single_process_dict, return_tensors="pt")
                processed_outputs.append(single_output)
                # 提取 pixel_values (假设 LLMprocessor 返回字典)
                # 确保 pixel_values 是单个样本的 tensor
                if "pixel_values" in single_output:
                     # .to(self.dev) 移动到 GPU
                    all_pixel_values.append(single_output["pixel_values"].to(self.dev))
                elif "pixel_values_videos" in single_output: # 处理视频可能的键名
                    all_pixel_values.append(single_output["pixel_values_videos"].to(self.dev))
                else:
                     print("Warning: 'pixel_values' or 'pixel_values_videos' not found in LLMprocessor output for sample", i)


            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # 可以选择跳过错误样本或引发异常
                continue # 跳过这个样本

        # 检查是否有成功处理的样本
        if not processed_outputs or not all_pixel_values:
             raise ValueError("No samples were processed successfully.")


        # 提取 input_ids 和 media_offset (假设它们在 processor 输出中)
        all_input_ids = [out['input_ids'].squeeze(0) for out in processed_outputs] # 移除批次维度 (1, seq_len) -> (seq_len)
        all_media_offsets = [out['media_offset'][0] for out in processed_outputs] # 假设 media_offset 也是批处理形式 (1, num_media) -> (num_media)


        # 2. 确定最大长度
        max_len = max(len(ids) for ids in all_input_ids)

        # 3. 手动填充 input_ids 和创建 attention_mask (左填充)
        padded_input_ids = []
        attention_masks = []
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # 如果 tokenizer 没有 pad_token_id，通常使用 eos_token_id
            pad_token_id = self.tokenizer.eos_token_id
            print(f"Warning: pad_token_id not set, using eos_token_id ({pad_token_id}) for padding.")
        if pad_token_id is None:
             raise ValueError("Tokenizer must have a pad_token_id or eos_token_id for padding.")


        for input_ids in all_input_ids:
            current_len = len(input_ids)
            padding_len = max_len - current_len

            # 创建填充张量
            padding_tensor = torch.full((padding_len,), pad_token_id, dtype=input_ids.dtype, device=self.dev)
            # 左填充 input_ids
            padded_ids = torch.cat([padding_tensor, input_ids.to(self.dev)], dim=0)
            padded_input_ids.append(padded_ids)

            # 创建 attention_mask (0 表示填充, 1 表示实际 token)
            mask = torch.cat([
                torch.zeros(padding_len, dtype=torch.long, device=self.dev),
                torch.ones(current_len, dtype=torch.long, device=self.dev)
            ], dim=0)
            attention_masks.append(mask)

        # 将列表堆叠成批处理张量
        batched_input_ids = torch.stack(padded_input_ids, dim=0)
        batched_attention_mask = torch.stack(attention_masks, dim=0)

        # 4. 处理 image_embeds (之前已收集 pixel_values)
        # 确保 all_pixel_values 中的张量形状兼容，然后合并
        # 注意：如果每个样本的图像/视频帧数不同，这里的 cat 可能需要更复杂的处理
        # 假设 self.LLM.forward_image 可以处理批处理的 pixel_values
        try:
            # 合并 pixel_values, 假设它们有兼容的形状 (除了批次维度)
            # 如果每个样本的帧数不同，这里会报错，需要特殊处理
            batched_pixel_values = torch.cat(all_pixel_values, dim=0)
        except RuntimeError as e:
             print(f"Error concatenating pixel_values: {e}. This might happen if samples have different numbers of frames.")
             # 这里需要根据你的数据和模型决定如何处理，例如填充帧或单独处理
             raise e


        with torch.no_grad():
            # 假设 forward_image 返回 (B*num_frames, embed_dim) 或类似结构
            # 你可能需要根据 forward_image 的输出和模型期望的输入调整 image_embeds 的形状
            image_embeds = self.LLM.forward_image(batched_pixel_values)
            # 可能需要 reshape 或调整 image_embeds 以匹配模型输入格式


        # 5. 处理 media_offset
        # media_offset 的处理取决于模型如何使用它。
        # 如果模型期望一个列表，其中每个元素是对应样本的偏移量张量，那么 all_media_offsets 可能已经足够。
        # 如果模型期望一个填充或堆叠的张量，你需要根据填充情况调整偏移量或进行填充。
        # 对于左填充，偏移量的值通常不需要改变，但它们现在是相对于填充后序列的索引。
        # 简单的处理方式是保持列表形式：
        final_media_offsets = all_media_offsets # 直接使用列表

        batched_inputs = {
            "input_ids": batched_input_ids,
            "attention_mask": batched_attention_mask,
            "media_offset": final_media_offsets,
            "pixel_values": batched_pixel_values,
        }

        return batched_inputs


    def forward(self, image_or_video=None, labels=None, **args):

        batched_inputs = self.processor(image_or_video)

        with torch.no_grad():
            image_embeds = self.LLM.forward_image(batched_inputs.pop("pixel_values"))
        
        outputs = self.HyperQwen2ForCausalLM_forward(
            image_embeds=image_embeds,
            labels=labels,
            **batched_inputs,
        )
        del batched_inputs, image_embeds
        torch.cuda.empty_cache()
    
        return outputs
    
    @property
    def dev(self):
        return self.LLM.device


class video_dataset(torch.utils.data.Dataset):
    def __init__(self, anno_file, data_prefix, phase, sample_types):
        super().__init__()

        self.video_infos = []
        self.phase = phase
        self.sample_types = sample_types
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.samplers = {}

        with open(anno_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split(",")
                filename, a, t, label = line_split
                label = float(a), float(t), float(label)

                filename = os.path.join(data_prefix, filename)
                self.video_infos.append(dict(filename=filename, label=label))

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        video = self.video_infos[idx]
        video_path = video["filename"]

        metadata = ffmpeg.probe(video_path)
        meta_stream = metadata["streams"][0]

        video_inf = {
            # "file size": f"{os.path.getsize(metadata["format"]["filename"])/1048576 :.2f}MB",
            # "duration": f"{float(meta_stream["duration"]):.0f}s",
            "resolution": f"{meta_stream['width']}x{meta_stream['height']}",
            "frame rate": f"{eval(meta_stream['avg_frame_rate']):.2f}fps",
            "bit rate": f"{int(meta_stream['bit_rate'])//1000}Kbps",
            "codec": meta_stream["codec_name"],
        }

        a, t, video_label = video["label"]
        video_frames = encode_video(video_path, num_frames=self.sample_types["clip_len"])
        video = {"info": video_inf, "data": video_frames}
        return video, video_label
    

class ImageJsonDataset(torch.utils.data.Dataset):
    def __init__(self, dir, anno_file):
        self.dir = dir

        with open(anno_file, 'r') as f:
            self.data = json.load(f)["files"]


    def __len__(self):
        # return 100
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.dir + "/" + self.data[idx]['image']
        label = self.data[idx]['score']
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        file_format = os.path.basename(img_path).split(".")[-1].upper()
        image_inf = {
            "Format": image.format if image.format else file_format,
            "File Size": f"{os.path.getsize(img_path)>>10:.0f}KB",
            "Resolution": f"{width}x{height}",
        }
        img = {"info": image_inf, "data": image}
        return img, label
    
    def collate_fn(self, batch):
        images, labels = zip(*batch)
        images = [img for img in images]
        labels = torch.tensor(labels, dtype=torch.float32)
        return images, labels


def uniform_sample(l, n, randomize=True):
    gap = len(l) / n
    if randomize:
        idxs = [int(i * gap + random.uniform(0, gap)) for i in range(n)]
        idxs = [min(i, len(l) - 1) for i in idxs]
    else:
        # uniform sampling
        idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def encode_video(video_path, num_frames=16, random_sampling=True):
    vr = VideoReader(video_path, ctx=cpu(0))
    base_fps_divisor = 6
    if random_sampling:
        sample_fps_divisor = base_fps_divisor + random.uniform(-1, 1)
        sample_fps = max(1, round(vr.get_avg_fps() / sample_fps_divisor))
    else:
        sample_fps = round(vr.get_avg_fps() / base_fps_divisor)
    
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > num_frames:
        frame_idx = uniform_sample(frame_idx, num_frames, randomize=random_sampling)
    
    frames = vr.get_batch(frame_idx).numpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


def cosine_weighted_loss(model, batch, val_embed, device):
    image_or_video = batch[0]
    labels = batch[1].to(device)

    outputs = model(image_or_video=image_or_video)
    logits = outputs.logits  # (bsz, seq_len, vocab_size)
    last_token_logits = logits[:, -1, :]  # (bsz, vocab_size)

    k = 100
    topk = torch.topk(last_token_logits, k, dim=-1)
    top_k_logits, top_k_indices = topk.values, topk.indices  # (bsz, k)

    embedding_layer = model.LLM.get_input_embeddings()
    top_k_embeddings = embedding_layer(top_k_indices)  # (bsz, k, dim)
    val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)  # (1,1,dim)

    weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)  # (bsz, k)
    weighted_logits = top_k_logits * weights  # (bsz, k)
    score = torch.sum(weighted_logits, dim=-1)  # (bsz,)

    loss = plcc_loss(score, labels)
    return loss, score, labels

def train_lora_cosine(
    model, 
    train_dataset, 
    val_dataset, 
    val_embed, 
    optimizer, 
    device, 
    epochs=5, 
    bsz=8, 
    writer=None
):
    from torch.utils.data import DataLoader
    import tqdm

    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for epoch in range(epochs):
        model.train()
        dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
        total_loss = 0
        pred_scores = []
        gt_scores = []
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            loss, score, labels = cosine_weighted_loss(model, batch, val_embed, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_scores.extend(score.detach().cpu().tolist())
            gt_scores.extend(labels.detach().cpu().tolist())
            if writer:
                writer.add_scalar("Loss/step", loss.item(), epoch * len(dataloader) + i)
            torch.cuda.empty_cache()
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(dataloader):.4f}")

        # 验证
        model.eval()              
        for nm, val_data in val_dataset.items():
            print(f"Validating on {nm} dataset...")  
            val_pred, val_gt = [], []
            val_loader = DataLoader(val_data, batch_size=bsz, shuffle=False, num_workers=2, collate_fn=val_data.collate_fn,drop_last=True)
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, desc="Validation"):
                    _, score, labels = cosine_weighted_loss(model, batch, val_embed, device)
                    val_pred.extend(score.squeeze().cpu().tolist())
                    val_gt.extend(labels.cpu().tolist())
            from scipy.stats import spearmanr, pearsonr
            srcc = spearmanr(val_pred, val_gt).statistic
            plcc = pearsonr(val_pred, val_gt).statistic
            print(f"Validation: SRCC={srcc:.4f}, PLCC={plcc:.4f}")
            if writer:
                writer.add_scalar(f"Validation/{nm}_SRCC", srcc, epoch)
                writer.add_scalar(f"Validation/{nm}_PLCC", plcc, epoch)



def trainq(model, val_datasets, optimizer, bsz=4):

    # 训练模型
    if model.linear_sw.bias.requires_grad:
        # get logits indices
        logits = []
        scores = []
        model.eval()
        with torch.no_grad():
            for name, data in val_datasets["train"].items():
                dataloader = DataLoader(data, batch_size=bsz, shuffle=True, num_workers=8, collate_fn=data.collate_fn)
                for batch in tqdm.tqdm(dataloader, ncols=100):
                    image_or_video = batch[0]
                    labels = batch[1].to(model.LLM.device)

                    outputs = model(image_or_video=image_or_video, labels=labels)
                    logits.append(outputs.logits[:,-1].cpu())
                    scores.append(labels.cpu())

            logits_all = torch.cat(logits)
            lmax = torch.max(logits_all, dim=0)
            top300max = torch.topk(lmax.values, 300, dim=-1)
            model.logi_indices = top300max.indices
            logits_top300 = logits_all[:,top300max.indices]
            scores_all = torch.cat(scores)


        # get logits weights linear_sw
        for param in model.linear_sw.parameters():
            param.requires_grad = True
            nn.init.constant_(param, 0.)
        logits_top300 = logits_top300.to(device).requires_grad_(True)  # 添加梯度要求
        scores_all = scores_all.to(device)
        optimizer_sw = torch.optim.Adam(model.linear_sw.parameters(), lr=1e-3)
        scheduler_sw = torch.optim.lr_scheduler.StepLR(optimizer_sw, step_size=100, gamma=0.5)
        for epoch_sw in tqdm.tqdm(range(200)):
            # 训练正则化的线性回归
            model.linear_sw.train()
            model.linear_sw.requires_grad_(True)
            optimizer_sw.zero_grad()
            pred_scores = model.linear_sw(logits_top300)
            l1_norm = sum(p.abs().sum() for p in model.linear_sw.parameters())
            loss = plcc_loss(pred_scores, scores_all.float())
            loss.backward()
            optimizer_sw.step()
            scheduler_sw.step()
            
            # 测试
            if epoch_sw % 30 != 0:
                continue
            model.linear_sw.eval()
            with torch.no_grad():
                pred_scores = model.linear_sw(logits_top300)
            pred_scores = pred_scores.squeeze(-1).detach().cpu().numpy()
            spearmanrcc, p_value = spearmanr(pred_scores, scores_all.detach().cpu().numpy())
            print("Spearmanr:", spearmanrcc, "P-value:", p_value)

        model.linear_sw.requires_grad_(False)
        # 保存模型参数
        sw_dict = {
            "linear_sw": model.linear_sw.state_dict(),  # 保存线性回归模型参数
            "logi_indices": model.logi_indices,          # 保存logits索引
        }
        torch.save(sw_dict, f"{run_dir}/linear_sw.pth")

        # 验证测试数据集初始指标, epoch标识为-1
        model.eval()
        for name, data in val_datasets["test"].items():
            spearmanrcc, pearsonrcc = evaluate(model, data, bsz=4)
            print(f"{name} eval Spearmanr: {spearmanrcc[0]:.4f}, Pearsonr: {pearsonrcc[0]:.4f}")


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    global_step = 0

    epochs = 10
    for epoch in range(epochs):
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)


        
        for name, data in val_datasets["train"].items():
            # 训练模型主体
            model.train()
            total_loss = 0
            pred_scores = []
            gt_scores = []
            dataloader = DataLoader(data, batch_size=bsz, shuffle=True, num_workers=8, collate_fn=dataset.list_collate)
            for i, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)):

                image_or_video = batch[0]
                labels = batch[1].to(model.LLM.device)

                outputs = model(image_or_video=image_or_video, labels=labels)
                plccloss, rankloss, score = outputs.loss
                
                # 记录loss和其他指标，不管是否更新都记录
                current_loss = plccloss.item()
                total_loss += current_loss
                
                pred_scores.extend(score.detach().cpu().tolist())
                gt_scores.extend(labels.detach().cpu().tolist())
                
                loss = plccloss  # 如果需要也可以加上rankloss: loss = plccloss + rankloss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                    
                # 记录到tensorboard
                writer.add_scalar("Loss/step", current_loss, global_step)
                global_step += 1

                # 清理不需要的变量
                del outputs, plccloss, rankloss, score
                if 'loss' in locals():
                    del loss
                torch.cuda.empty_cache()  # 定期清理GPU缓存

            writer.add_scalar(f"train Loss/epoch", total_loss, epoch)
            print(f"Epoch {epoch+1} Average Loss: {total_loss:.4f}")

            pred_scores = torch.tensor(pred_scores)
            gt_scores = torch.tensor(gt_scores)
            spearmanrcc, pearsonrcc = spearmanr(pred_scores, gt_scores), pearsonr(pred_scores, gt_scores)
            writer.add_scalar("Spearmanr/train", spearmanrcc[0], epoch)
            writer.add_scalar("Pearsonr/train", pearsonrcc[0], epoch)
            print(f"train Spearmanr: {spearmanrcc[0]:.4f}, Pearsonr: {pearsonrcc[0]:.4f}")

        try:
            # 创建一个字典来存储所有需要保存的参数
            save_dict = {
                'optimizer_state': optimizer.state_dict(),  # 保存优化器状态
                'scheduler_state': scheduler.state_dict(),  # 保存调度器状态
                'epoch': epoch,                            # 保存当前epoch
            }
            
            # 保存prompt参数（如果启用）
            if hasattr(model, 'prompt_embeddings') and model.trainable_prompt:
                save_dict['prompt_embeddings'] = model.prompt_embeddings.state_dict()
            
            # 保存LoRA参数（如果启用）
            if hasattr(model, 'lora_r') and model.lora_r > 0:
                save_dict['lora_params'] = {}
                # 遍历并保存所有LoRA参数
                for name, module in model.named_modules():
                    if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                        # 对于PEFT库的LoRA实现，需要提取特定状态
                        save_dict['lora_params'][name] = module.state_dict()
            
            # 保存为单个文件
            torch.save(save_dict, f"{run_dir}/model_epoch_{epoch}.pth")
            print(f"model saved at epoch {epoch+1} in {run_dir}")
        except Exception as e:
            print(f"保存失败: {e}")
    
    
        # 验证
        model.eval()
        for name, data in val_datasets["test"].items():
            spearmanrcc, pearsonrcc = evaluate(model, data, bsz=4)
            print(f"{name} eval Spearmanr: {spearmanrcc[0]:.4f}, Pearsonr: {pearsonrcc[0]:.4f}")
            writer.add_scalar(f"Spearmanr/val-{name}", spearmanrcc[0], epoch)
            writer.add_scalar(f"Pearsonr/val-{name}", pearsonrcc[0], epoch)
 
        scheduler.step()
    writer.close()


def evaluate(model, val_dataset, bsz=4):
    # 验证
    model.eval()
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=dataset.list_collate)
    
    with torch.no_grad():
        val_pred_scores = []
        val_gt_scores = []
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"Validation", ncols=100)):

            image_or_video = batch[0]
            labels = batch[1].to(model.LLM.device)

            outputs = model(image_or_video=image_or_video, labels=labels)
            plccloss, rankloss, score = outputs.loss
            val_pred_scores += (score.cpu().tolist())
            val_gt_scores += (labels.cpu().tolist())
        
            # 清理不需要的变量
            del batch, image_or_video, labels, outputs, plccloss, rankloss, score
            torch.cuda.empty_cache()  # 定期清理GPU缓存
        
        val_pred_scores = torch.tensor(val_pred_scores)
        val_gt_scores = torch.tensor(val_gt_scores)
        spearmanrcc = spearmanr(val_pred_scores, val_gt_scores)
        pearsonrcc = pearsonr(val_pred_scores, val_gt_scores)
    return spearmanrcc, pearsonrcc


class aigc_video_dataset(torch.utils.data.Dataset):
    def __init__(self, anno_file, data_prefix, phase, sample_types):
        super().__init__()

        self.data_prefix = data_prefix
        self.sample_types = sample_types
        self.samplers = {}

        # 使用pandas库读取csv
        self.df = pd.read_csv(anno_file)

    def __len__(self):
        return len(self.df)
        return 100

    def __getitem__(self, idx):
        video = self.df.iloc[idx]
        video_path = os.path.join(self.data_prefix, video["video_name"])
        prompt = video["Prompt"]
        MOS = video.get("Overall_MOS", 0.0)
        metadata = ffmpeg.probe(video_path)
        meta_stream = metadata["streams"][0]

        video_inf = {
            # "file size": f"{os.path.getsize(metadata["format"]["filename"])/1048576 :.2f}MB",
            # "duration": f"{float(meta_stream["duration"]):.0f}s",
            "resolution": f"{meta_stream['width']}x{meta_stream['height']}",
            # "frame rate": f"{eval(meta_stream['avg_frame_rate']):.2f}fps",
            # "bit rate": f"{int(meta_stream['bit_rate'])//1000}Kbps",
            # "codec": meta_stream["codec_name"],
            "prompt": prompt,
            "video_name": video["video_name"],
        }

        video_frames = encode_video(video_path, num_frames=self.sample_types["clip_len"])
        video = {"info": video_inf, "data": video_frames, "MOS": MOS}
        return video
    
    def collate_fn(self, batch):
        # 批处理函数
        video_inf = [item["info"] for item in batch]
        video_frames = [item["data"] for item in batch]
        video = {"info": video_inf, "data": video_frames}
        label = torch.tensor([item["MOS"] for item in batch])
        return video, label



if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    date_str = time.strftime("%Y-%m-%d", time.localtime())
    time_str = time.strftime("%H:%M:%S", time.localtime())
    run_dir = f"runs/{date_str}/{time_str}"
    writer = SummaryWriter(run_dir)
    import yaml
    data_yml = "iqa.yml"
    
    opt = yaml.safe_load(open(data_yml, "r"))
    val_datasets = {}
    for phase, datasets in opt.items():
        if phase not in ["train", "test"]:
            continue
        val_datasets[phase] = {}
        for name, data_args in datasets.items():
            val_datasets[phase][name] = globals().get(data_args["type"])(**data_args["args"])


    model = QualityOwl3Model(new_layers=None, lora_r=16, trainable_prompt=False).to(device)
    model.linear_sw.requires_grad_(True)

    # train_dict = torch.load("runs/2025-04-17/17:36:38/linear_sw.pth", map_location=device)
    # model.linear_sw.load_state_dict(train_dict["linear_sw"])
    # model.logi_indices = train_dict["logi_indices"]
    # model.linear_sw.requires_grad_(False)

    # 收集所有需要优化的参数
    params_to_optimize = []
    
    # 添加所有LoRA参数
    if hasattr(model, 'lora_r') and model.lora_r > 0:
        # 查找所有LoRA模块
        for name, param in model.named_parameters():
            # 根据参数名判断是否为LoRA参数（通常包含"lora_"字符串）
            if 'lora_' in name and param.requires_grad:
                lora_params = {'params': param, 'lr': 1e-4}  # 可以为LoRA使用不同的学习率
                params_to_optimize.append(lora_params)

    # 除了要优化的参数，其余参数全部冻结
    for name, param in model.LLM.named_parameters():
        if 'lora_' not in name and param.requires_grad:
            param.requires_grad = False

    
    # 创建AdamW优化器，使用收集到的所有参数
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # trainq(model, val_datasets, optimizer, bsz=8)
    from owl3_zeroshot import get_embed
    val_embed = get_embed(model, device)
    for nm, ds in val_datasets["train"].items():
        train_lora_cosine(
            model, 
            ds, 
            val_datasets["test"], 
            val_embed, 
            optimizer, 
            device, 
            epochs=10, 
            bsz=4, 
            writer=writer
        )


