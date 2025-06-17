from scipy import optimize
import tqdm
import torch, sys, transformers
import torch.nn as nn
import torch.nn.functional as F
import dataset
import ffmpeg
from decord import VideoReader, cpu
from PIL import Image
import random
import os,json
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1" # 去除颜色编码
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["AV_LOG_LEVEL"] = "quiet"  
os.environ["FFMPEG_LOGLEVEL"] = "quiet" 


from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import dataset
import yaml
import tqdm, time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device: ", device)

date_str = time.strftime("%Y-%m-%d", time.localtime())
time_str = time.strftime("%H:%M:%S", time.localtime())
run_dir = f"runs/{date_str}/{time_str}"
writer = SummaryWriter(run_dir)

class MultimodalQualityEvaluator(nn.Module):
    def __init__(
        self,
        task="IQA",
        model_path = "iic/mPLUG-Owl3-7B-241101",

    ): 
        super().__init__()
        self.task = task
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )

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


    def forward(self, image_or_video=None, labels=None, **args):

        batched_inputs = self.processor(image_or_video)

        with torch.no_grad():
            image_embeds = self.LLM.forward_image(batched_inputs.pop("pixel_values"))
        
        outputs = self.LLM.language_model(
            image_embeds=image_embeds,
            labels=labels,
            **batched_inputs,
        )
        del batched_inputs, image_embeds
        torch.cuda.empty_cache()
    
        return outputs
   

    def processor(self, data_and_info):
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
                    "content": f"{media_token[media_type]}Taking into account the details and the rationality of the {media_type[:-1]}, how would you rate the quality of this {media_type[:-1]}?",
                    # "content": f"{media_token[media_type]}The infomation of the {media_type[:-1]} is as follows:{media_info[i]}, how would you rate the quality of this {media_type[:-1]}?",
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
    

class ImageCsvDataset(torch.utils.data.Dataset):
    def __init__(self, dir, anno_file, image_key, score_key):
        super().__init__()
        self.dir = dir
        # 用pandas读取csv文件
        df = pd.read_csv(anno_file)
        self.data = df[[image_key, score_key]].values.tolist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.data[idx][0])
        label = self.data[idx][1]
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        file_format = os.path.basename(img_path).split(".")[-1].upper()
        image_inf = {
            "name": os.path.basename(img_path),
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


class TopkDataset(torch.utils.data.Dataset):
    def __init__(self, topk_data):
        super().__init__()
        self.topk_data = topk_data

    def __len__(self):
        return len(self.topk_data["logits"])

    def __getitem__(self, idx):
        logits = self.topk_data["logits"][idx]
        indices = self.topk_data["indices"][idx]
        gt_score = self.topk_data["gt_scores"][idx]
        return logits, indices, gt_score
    
    def collate_fn(self, batch):
        logits, indices, gt_scores = zip(*batch)
        logits = torch.stack(logits, dim=0)
        indices = torch.stack(indices, dim=0)
        gt_scores = torch.tensor(gt_scores, dtype=torch.float32)
        return logits, indices, gt_scores


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


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def embed_fit(model, val_dataset, val_embed, bsz=8, data_path=None):
    model.eval()
    if data_path is not None and os.path.exists(data_path):
        topk_data = torch.load(data_path)
        topk_dataset = TopkDataset(topk_data)
        valdataloader = DataLoader(topk_dataset, batch_size=bsz*5, shuffle=False, collate_fn=topk_dataset.collate_fn)
    else:
        valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=dataset.list_collate)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    val_gt_scores = []
    topk_data_logits = []
    topk_data_indices = []
    for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"fitting", ncols=100)):
        k = 300

        if data_path is None or not os.path.exists(data_path):

            image_or_video = batch[0]
            labels = batch[1]

            with torch.no_grad():
                outputs = model(image_or_video=image_or_video)
                logits = outputs.logits

            last_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            topk = torch.topk(last_token_logits, k, dim=-1) # Shapes: (batch_size, k)
            top_k_logits, top_k_indices = topk.values, topk.indices
        
            topk_data_logits.append(top_k_logits.cpu())
            topk_data_indices.append(top_k_indices.cpu())

        else:
            top_k_logits, top_k_indices, labels = batch


        embedding_layer = model.LLM.get_input_embeddings()
        top_k_embeddings = embedding_layer(top_k_indices.to(model.LLM.device))
        val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)

        weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
        weighted_logits = top_k_logits.to(model.LLM.device) * weights # Shape: (batch_size, k)
        score = torch.sum(weighted_logits, dim=-1) # Shape: (batch_size,)

        loss = plcc_loss(score, labels.to(score.device))
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
        val_gt_scores.extend(labels.cpu().tolist())

    if len(topk_data_logits) > 0:
        topk_data = {
            "logits": torch.cat(topk_data_logits, dim=0),
            "indices": torch.cat(topk_data_indices, dim=0),
            "gt_scores": torch.tensor(val_gt_scores),
        }

        return val_embed, topk_data 
    
    return val_embed, {}


def evaluate(model, val_dataset, val_embed, bsz=8):
    # 验证
    model.eval()
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=dataset.list_collate)
    
    with torch.no_grad():
        val_pred_scores = []
        val_gt_scores = []
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"Validation", ncols=100)):

            image_or_video = batch[0]
            labels = batch[1]

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

            # 6. 处理相似度权重 (根据您之前的逻辑)
            weights_sign = torch.sign(weights)
            # 将绝对值小于阈值的权重置零
            weights_sign[torch.abs(weights) < 1e-3] = 0.0
            # 乘以一个缩放因子 (这个因子是根据您之前的代码)
            processed_weights = weights_sign * 0.125

            # 7. 使用处理后的权重对 top_k_logits 进行加权
            weighted_logits = top_k_logits.to(device) * processed_weights # 形状: (batch_size, k)

            # 8. 将加权后的 logits 按样本求和，得到最终的质量分数
            score = torch.sum(weighted_logits, dim=-1) # 形状: (batch_size,)

            val_pred_scores.extend(score.cpu().tolist())
            val_gt_scores.extend(labels.cpu().tolist())
        
        val_pred_scores = torch.tensor(val_pred_scores)
        val_gt_scores = torch.tensor(val_gt_scores)
        
        spearmanrcc = spearmanr(val_pred_scores, val_gt_scores)
        pearsonrcc = pearsonr(val_pred_scores, val_gt_scores)


    return spearmanrcc, pearsonrcc, val_pred_scores, val_gt_scores


def q_evaluate(model, val_dataset, val_embed, bsz=2):
    # 验证
    model.eval()
    qbench_tokens = [model.tokenizer.encode(t)[0] for t in ["good", "poor"]]
    qalign_tokens = [model.tokenizer.encode(t)[0] for t in ["bad", "poor", "fair", "good", "excellent"]]
    
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=dataset.list_collate)
    
    with torch.no_grad():
        val_pred_scores = []
        qbench_val_scores = []
        qalign_val_scores = []
        val_gt_scores = []
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"Validation", ncols=100)):

            image_or_video = batch[0]
            labels = batch[1]
            val_gt_scores.extend(labels.cpu().tolist())

            outputs = model(image_or_video=image_or_video)
            logits = outputs.logits
            
            ###################### NEW METHOD #######################
            last_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            k = 300
            topk = torch.topk(last_token_logits, k, dim=-1) # Shapes: (batch_size, k)
            top_k_logits, top_k_indices = topk.values, topk.indices

            embedding_layer = model.LLM.get_input_embeddings()
            top_k_embeddings = embedding_layer(top_k_indices)
            val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)

            weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
            weighted_logits = top_k_logits.to(weights.device) * weights # Shape: (batch_size, k)
            score = torch.sum(weighted_logits, dim=-1) # Shape: (batch_size,)

            val_pred_scores.extend(score.cpu().tolist())
            ###################### NEW METHOD #######################
            
            ######################## Q-BENCH ########################
            binary_logits = last_token_logits[:, qbench_tokens] # Shape: (batch_size, 2)

            binary_probality = torch.softmax(binary_logits, dim=-1) # Shape: (batch_size, 2)
            q_bench_score = binary_probality[:, 0]

            qbench_val_scores.extend(q_bench_score.cpu().tolist())
            ######################## Q-BENCH ########################

            ######################## Q-ALIGN ########################
            target_logits = last_token_logits[:, qalign_tokens] # Shape: (batch_size, len(target_words))
            target_probality = torch.softmax(target_logits, dim=-1) # Shape: (batch_size, len(target_words))
            target_scores = torch.sum(target_probality * torch.tensor([1, 2, 3, 4, 5], device=target_probality.device), dim=-1) # Shape: (batch_size,)

            qalign_val_scores.extend(target_scores.cpu().tolist())
            ######################## Q-ALIGN ########################

        val_pred_scores = torch.tensor(val_pred_scores)
        qbench_val_scores = torch.tensor(qbench_val_scores)
        qalign_val_scores = torch.tensor(qalign_val_scores)
        val_gt_scores = torch.tensor(val_gt_scores)
        
        spearmanrcc = spearmanr(val_pred_scores, val_gt_scores).statistic
        pearsonrcc = pearsonr(val_pred_scores, val_gt_scores).statistic
        qbench_spearmanrcc = spearmanr(qbench_val_scores, val_gt_scores).statistic
        qbench_pearsonrcc = pearsonr(qbench_val_scores, val_gt_scores).statistic
        qalign_spearmanrcc = spearmanr(qalign_val_scores, val_gt_scores).statistic
        qalign_pearsonrcc = pearsonr(qalign_val_scores, val_gt_scores).statistic

        return_dict = {
            "srcc": spearmanrcc,
            "plcc": pearsonrcc,
            "qbench_srcc": qbench_spearmanrcc,
            "qbench_plcc": qbench_pearsonrcc,
            "qalign_srcc": qalign_spearmanrcc,
            "qalign_plcc": qalign_pearsonrcc,
        }


    return return_dict


if __name__ == "__main__":
    TASK = "IQA"

    YML_FILE = {
        "IQA": "iqa.yml",
        "IAA": "iaa.yml",
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

    model = MultimodalQualityEvaluator(TASK, model_path="iic/mPLUG-Owl3-7B-241101").to(device)

    text_dict = {   
        "IQA" : 
        {
            "goodtext" : " perfect superb outstanding excellent fantastic stunning phenomenal brilliant magnificent amazing remarkable beautiful awesome breathtaking great good decent fine sharp clear suitable vibrant rich vivid bright colorful",
            "badtext" : " bad terrible awful poor horrible disappointing unacceptable inadequate deficient blurry fuzzy compromised chaotic distorted weak mediocre sub lacking unclear dark noisy low problematic insufficient"
        },
        "IAA" : 
        {
            "goodtext": " beautiful stunning enchanting harmonious artistic pleasing exquisite stunning elegant graceful balanced vibrant evocative poignant serene sublime picturesque appealing striking gorgeous charming delightful sophisticated",
            "badtext": " mediocre poorly dull bland chaotic disple lacking amateur overly sub monotonous average clutter uninspired unpleasant discord garish mundane tacky glaring simplistic flat"
        }
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

    val_embed = embeddings["goodtext"].mean((0,1))-embeddings["badtext"].mean((0,1))

    # for name, val_dataset in val_datasets["train"].items():
    #     pre_save_path = f"exps/topk/{name}.pt" if os.path.exists(f"exps/topk/{name}.pt") else None
    #     val_embed, topk_data = embed_fit(model, val_dataset, val_embed, bsz=8, data_path=pre_save_path)
    #     torch.save(topk_data, f"exps/topk/{name}.pt") if pre_save_path is None else None

    # for name, val_dataset in val_datasets["test"].items():
    #     srcc, plcc, val_pred_scores, val_gt_scores = evaluate(model, val_dataset, val_embed, bsz=8)
    #     print(f"{name} srcc: {srcc.statistic}, plcc: {plcc.statistic}")
        
    for name, val_dataset in val_datasets["test"].items():
        results = q_evaluate(model, val_dataset, val_embed, bsz=1)
        print(f"********{name} results:*********")
        for key, value in results.items():
            print(f"{key}:\t {value}")

    # # 使用TopkDataset进行评估
    # for name, val_dataset in val_datasets["test"].items():
    #     pre_save_path = f"exps/topk/{name}.pt"
    #     if os.path.exists(pre_save_path):
    #         topk_data = torch.load(pre_save_path)
    #         topk_dataset = TopkDataset(topk_data)
    #         valdataloader = DataLoader(topk_dataset, batch_size=100, shuffle=False, collate_fn=topk_dataset.collate_fn)
            
    #         val_pred_scores = []
    #         val_gt_scores = []
            
    #         with torch.no_grad():
    #             for batch in tqdm.tqdm(valdataloader, desc=f"Evaluating {name}", ncols=100):
    #                 top_k_logits, top_k_indices, labels = batch
                    
    #                 embedding_layer = model.LLM.get_input_embeddings()
    #                 top_k_embeddings = embedding_layer(top_k_indices.to(model.LLM.device))
    #                 val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)
                    
    #                 weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
    #                 weighted_logits = top_k_logits.to(model.LLM.device) * weights
    #                 score = torch.sum(weighted_logits, dim=-1)
                    
    #                 val_pred_scores.extend(score.cpu().tolist())
    #                 val_gt_scores.extend(labels.cpu().tolist())
            
    #         val_pred_scores = torch.tensor(val_pred_scores)
    #         val_gt_scores = torch.tensor(val_gt_scores)
            
    #         srcc = spearmanr(val_pred_scores, val_gt_scores)
    #         plcc = pearsonr(val_pred_scores, val_gt_scores)
            
    #         print(f"{name} (TopkDataset) srcc: {srcc.statistic}, plcc: {plcc.statistic}")
    #     else:
    #         print(f"Topk data file not found for {name}: {pre_save_path}")

    # TODO 画出q align和q bench方案的散点图和分布图，判断是否不符合两边低，中间高的趋势