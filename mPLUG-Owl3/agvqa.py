import os
import yaml
import tqdm
import torch
import ffmpeg
import dataset
from PIL import Image
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import torch.nn.functional as F

from owl3_zeroshot import MultimodalQualityEvaluator, encode_video, plcc_loss

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
        video = {"info": video_inf, "data": video_frames}
        return video
    
    def collate_fn(self, batch):
        # 批处理函数
        video_inf = [item["info"] for item in batch]
        video_frames = [item["data"] for item in batch]
        video = {"info": video_inf, "data": video_frames}
        return video, []


def evaluate(model, val_dataset, val_embed, bsz=8):
    # 验证
    model.eval()
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn)
    
    with torch.no_grad():
        val_pred_scores = {}
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"Validation", ncols=100)):

            image_or_video = batch[0]
            video_names = [item["video_name"] for item in image_or_video["info"]]
            outputs = model(image_or_video=image_or_video)
            logits = outputs.logits

            # 1. 获取最后一个 token 的 logits
            last_token_logits = logits[:, -1, :]  # 形状: (batch_size, vocab_size)

            # 2. 获取 top-k 的 logits 和对应的 token 索引
            topk_output = torch.topk(last_token_logits, 300, dim=-1)
            top_k_logits = topk_output.values    # 形状: (batch_size, k)
            top_k_indices = topk_output.indices  # 形状: (batch_size, k)

            # 3. 获取模型的词嵌入层
            embedding_layer = model.LLM.get_input_embeddings()

            # 4. 将 top-k 索引转换为词嵌入向量
            top_k_embeddings = embedding_layer(top_k_indices.to(device))  # 形状: (batch_size, k, embedding_dim)

            # 5. 计算 top-k 词嵌入与 val_embed 之间的余弦相似度
            #    需要调整 val_embed 的形状以进行广播
            val_embed_unsqueezed = val_embed.to(device).unsqueeze(0).unsqueeze(0)
            # val_embed_unsqueezed 形状: (1, 1, embedding_dim)

            weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1) # 形状: (batch_size, k)

            # 7. 使用处理后的权重对 top_k_logits 进行加权
            weighted_logits = top_k_logits.to(device) * weights # 形状: (batch_size, k)

            # 8. 将加权后的 logits 按样本求和，得到最终的质量分数
            score = torch.sum(weighted_logits, dim=-1) # 形状: (batch_size,)

            # 将video_names和score存储到字典中
            for j in range(len(video_names)):
                video_name = video_names[j]
                val_pred_scores[video_name] = score[j].item()

        # val_pred_scores存储为csv文件，列名video_name, Overall_MOS
        


    return val_pred_scores


class aigv_evaluator(MultimodalQualityEvaluator):
    def processor(self, data_and_info):
        image_or_video = data_and_info["data"]
        media_info = data_and_info["info"]
        prompts = [inf["prompt"] for inf in media_info]
        media_type = "images" if isinstance(image_or_video[0], Image.Image) else "videos"
        batch_size = len(image_or_video)
        media_token = {"images": "<|image|>", "videos": "<|video|>"}

        # 准备批处理数据 (保持不变)
        batched_messages = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "user",
                    "content": f"{media_token[media_type]}The infomation of the {media_type[:-1]} is as follows:{media_info[i]}, The prompt used to genetate this video is {prompts[i]}.Taking into account the details and the rationality of the {media_type[:-1]}, how would you rate the quality of this {media_type[:-1]}?",
                    # "content": f"{media_token[media_type]}The infomation of the {media_type[:-1]} is as follows:{media_info[i]}, how would you rate the quality of this {media_type[:-1]}?",
                },
                {"role": "assistant", "content": f"The quality of the {media_type[:-1]} is very"},
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

            # 假设 LLMprocessor 返回一个包含 'input_ids', 'media_offset', 'pixel_values' 的字典或对象
            # 确保返回的是 tensor，至少 pixel_values 需要是
            single_output = self.LLMprocessor(**single_process_dict, return_tensors="pt")
            processed_outputs.append(single_output)
            # 提取 pixel_values (假设 LLMprocessor 返回字典)
            # 确保 pixel_values 是单个样本的 tensor
            if "pixel_values" in single_output:
                all_pixel_values.append(single_output["pixel_values"].to(self.dev))
            elif "pixel_values_videos" in single_output: # 处理视频可能的键名
                all_pixel_values.append(single_output["pixel_values_videos"].to(self.dev))
            else:
                    print("Warning: 'pixel_values' or 'pixel_values_videos' not found in LLMprocessor output for sample", i)

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


def fit_val_embed(train_dataset, model, val_embed, bsz=4):
    # 训练集拟合 val_embed
    model.eval()
    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    
    # 将 val_embed 转换为可优化的参数
    val_embed = torch.nn.Parameter(val_embed.clone().detach().requires_grad_(True))
    
    optimizer = torch.optim.Adam([val_embed], lr=1e-2)
    criterion = plcc_loss
    
    # 缓存模型输出的 topk_output
    cached_topk_outputs = []
    cached_labels = []
    
    for epoch in range(3):  # 训练3个epoch
        if epoch == 0:
            # 第一轮：记录模型输出的topk_output
            print("第一轮：缓存模型输出...")
            for i, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Caching model outputs", ncols=100)):
                image_or_video = batch[0]
                with torch.no_grad():
                    # 通过模型获取 logits
                    outputs = model(image_or_video=image_or_video)
                    logits = outputs.logits

                # 1. 获取最后一个 token 的 logits
                last_token_logits = logits[:, -1, :]
                # 2. 获取 top-k 的 logits 和对应的 token 索引
                topk_output = torch.topk(last_token_logits, 300, dim=-1)
                
                # 缓存 topk_output（转移到CPU以节省GPU内存）
                cached_topk_outputs.append({
                    'values': topk_output.values.cpu(),
                    'indices': topk_output.indices.cpu()
                })
                
                # 缓存对应的标签
                batch_labels = torch.tensor(
                    train_dataset.df["Overall_MOS"].values[i*bsz:(i+1)*bsz], 
                    dtype=torch.float32
                )[:len(topk_output.values)]  # 确保长度匹配
                cached_labels.append(batch_labels)
        
        # 使用缓存的数据进行训练
        print(f"第{epoch+1}轮训练...")
        for i, cached_data in enumerate(tqdm.tqdm(cached_topk_outputs, desc=f"Training epoch {epoch+1}", ncols=100)):
            # 从缓存中获取数据
            top_k_logits = cached_data['values'].to(device)     # 形状: (batch_size, k)
            top_k_indices = cached_data['indices'].to(device)   # 形状: (batch_size, k)
            labels = cached_labels[i].to(device)
            
            # 3. 获取模型的词嵌入层
            embedding_layer = model.LLM.get_input_embeddings()
            # 4. 将 top-k 索引转换为词嵌入向量
            top_k_embeddings = embedding_layer(top_k_indices)  # 形状: (batch_size, k, embedding_dim)
            # 5. 计算 top-k 词嵌入与 val_embed 之间的余弦相似度
            val_embed_unsqueezed = val_embed.unsqueeze(0).unsqueeze(0)
            # val_embed_unsqueezed 形状: (1, 1, embedding_dim)
            weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1) # 形状: (batch_size, k)
            # 6. 使用处理后的权重对 top_k_logits 进行加权
            weighted_logits = top_k_logits * weights # 形状: (batch_size, k)
            # 7. 将加权后的 logits 按样本求和，得到最终的质量分数
            score = torch.sum(weighted_logits, dim=-1) # 形状: (batch_size,)
            # 计算损失
            loss = criterion(score, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 返回拟合后的 val_embed
    return val_embed.detach()


if __name__ == "__main__":
    TASK = "AIGV"

    YML_FILE = {
        "IQA": "iqa.yml",
        "IAA": "iaa.yml",
        "AIGV": "agvqa.yml"
    }

    data_yml = YML_FILE[TASK]
    
    opt = yaml.safe_load(open(data_yml, "r"))
    val_datasets = {}
    for phase, datasets in opt.items():
        if phase not in ["train", "test"]:
            continue
        val_datasets[phase] = {}
        for name, data_args in datasets.items():
            val_datasets[phase][name] = globals().get(data_args["type"])(**data_args["args"])

    model = aigv_evaluator(TASK, model_path="iic/mPLUG-Owl3-7B-241101").to(device)

    text_dict = {   
        "IQA" : 
        {
            "positive" : " perfect superb outstanding excellent fantastic stunning phenomenal brilliant magnificent amazing remarkable beautiful awesome breathtaking great good decent fine sharp clear suitable vibrant rich vivid bright colorful",
            "negative" : " bad terrible awful poor horrible disappointing unacceptable inadequate deficient blurry fuzzy compromised chaotic distorted weak mediocre sub lacking unclear dark noisy low problematic insufficient"
        },
        "IAA" : 
        {
            "positive": " beautiful stunning enchanting harmonious artistic pleasing exquisite stunning elegant graceful balanced vibrant evocative poignant serene sublime picturesque appealing striking gorgeous charming delightful sophisticated",
            "negative": " mediocre poorly dull bland chaotic disple lacking amateur overly sub monotonous average clutter uninspired unpleasant discord garish mundane tacky glaring simplistic flat"
        },
        "AIGV" :
        {
            "positive" : " perfect superb outstanding excellent fantastic stunning phenomenal brilliant magnificent amazing remarkable beautiful awesome breathtaking great good decent fine sharp clear suitable vibrant rich vivid bright colorful",
            "negative" : " bad terrible awful poor horrible disappointing unacceptable inadequate deficient blurry fuzzy compromised chaotic distorted weak mediocre sub lacking unclear dark noisy low problematic insufficient"
        },
    }
    embeddings={}
    for name,words in text_dict[TASK].items():
        # 通过 tokenizer 得到 input_ids
        inputs = model.tokenizer(words, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]  # shape: (1, sequence_length)

        # 从模型中提取嵌入层
        embedding_layer = model.LLM.get_input_embeddings()

        # 用嵌入层嵌入 input_ids
        embeddings[name] = embedding_layer(input_ids)

    val_embed = embeddings["positive"].mean((0,1))-embeddings["negative"].mean((0,1))

    # 使用训练集拟合一个val_embed
    for name, train_dataset in val_datasets["train"].items():
        val_embed = fit_val_embed(train_dataset, model, val_embed, bsz=4)


    for name, val_dataset in val_datasets["test"].items():
        evaluate(model, val_dataset, val_embed, bsz=4)
