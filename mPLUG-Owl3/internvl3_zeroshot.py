from scipy import optimize
import tqdm
import torch, sys, transformers
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
import random
import os,json
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1" # 去除颜色编码
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["AV_LOG_LEVEL"] = "quiet"  
os.environ["FFMPEG_LOGLEVEL"] = "quiet" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import yaml
import tqdm, time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

def list_collate(batch):
    data_items = [item[0] for item in batch] # item[0] is the data dict {"info":..., "data":...}
    labels = [item[1] for item in batch]     # item[1] is the score
    return data_items, torch.tensor(labels, dtype=torch.float32)

class MultimodalQualityEvaluator(nn.Module):
    def __init__(
        self,
        task="IQA",
        model_path="OpenGVLab/InternVL3-8B-hf",
        local_files_only=False,
    ): 
        super().__init__()
        self.task = task
        
        # 确定合适的数据类型
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
            print("Warning: CUDA not available. Using CPU with torch.float32 for InternVL3 model.")
        
        print(f"InternVL3 model will be loaded with dtype: {dtype}")

        # 加载InternVL3模型
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        
        # 加载处理器
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        
        self.tokenizer = self.processor.tokenizer

    def _prepare_messages(self, data_item):
        # data_item: {"data": PIL_Image_or_frames_list, "info": {...}}
        media_data = data_item["data"]
        is_video = isinstance(media_data, list) and all(isinstance(frame, Image.Image) for frame in media_data)
        media_type_str = "video" if is_video else "image"

        if is_video:
            # 对于视频，InternVL3期望一个视频URL或本地路径
            # 但由于我们有帧列表，我们需要特殊处理
            # 这里我们暂时使用第一帧作为代表，或者您可以保存为临时视频文件
            content_item = {
                "type": "image",  # 暂时使用image处理视频帧
                "url": media_data[0] if len(media_data) > 0 else None,  # 使用第一帧
            }
        else:
            # 对于图像
            content_item = {
                "type": "image",
                "url": media_data,  # PIL Image
            }

        if self.task == "IQA":
            user_text = f"How would you rate the quality of this {media_type_str}?"
            assistant_prefix = f"The quality of this {media_type_str} is very"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        content_item,
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        elif self.task == "IAA":
            user_text = f"Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this {media_type_str}?"
            assistant_prefix = f"The {media_type_str} is"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        content_item,
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return messages, assistant_prefix

    def processor_batch(self, data_and_info_batch):
        # data_and_info_batch: List of {"data": PIL_Image_or_frames_list, "info": {...}}
        
        batch_messages = []
        batch_assistant_prefixes = []

        for item in data_and_info_batch:
            messages, assistant_prefix = self._prepare_messages(item)
            batch_messages.append(messages)
            batch_assistant_prefixes.append(assistant_prefix)

        # 为批处理创建输入
        try:
            # 首先处理原始消息
            inputs = self.processor.apply_chat_template(
                batch_messages,
                padding=True, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            )
            
            # 然后为每个样本添加assistant前缀
            batch_size = len(batch_assistant_prefixes)
            new_input_ids = []
            new_attention_masks = []
            
            for i in range(batch_size):
                # 获取当前样本的有效长度（排除padding）
                current_input_ids = inputs['input_ids'][i]
                current_attention_mask = inputs['attention_mask'][i] if 'attention_mask' in inputs else None
                
                # 找到实际内容的结束位置（非padding部分）
                if current_attention_mask is not None:
                    valid_length = current_attention_mask.sum().item()
                    actual_input_ids = current_input_ids[:valid_length]
                else:
                    # 如果没有attention_mask，假设没有padding
                    actual_input_ids = current_input_ids
                    valid_length = len(actual_input_ids)
                
                # 编码assistant前缀
                prefix_tokens = self.processor.tokenizer.encode(
                    batch_assistant_prefixes[i], 
                    add_special_tokens=False
                )
                prefix_tensor = torch.tensor(prefix_tokens, dtype=torch.long)  # 确保是long类型
                
                # 拼接实际内容和前缀
                new_input_ids_sample = torch.cat([
                    actual_input_ids, 
                    prefix_tensor.to(current_input_ids.device)
                ])
                
                new_input_ids.append(new_input_ids_sample)
                
                # 创建对应的attention_mask
                new_attention_mask = torch.ones(
                    len(new_input_ids_sample), 
                    dtype=torch.long  # 确保是long类型
                )
                new_attention_masks.append(new_attention_mask)
            
            # 找到新的最大长度
            max_new_length = max(len(ids) for ids in new_input_ids)
            
            # 重新padding到统一长度
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0
            
            padded_input_ids = []
            padded_attention_masks = []
            
            for i in range(batch_size):
                current_length = len(new_input_ids[i])
                padding_length = max_new_length - current_length
                
                if padding_length > 0:
                    # 左padding
                    pad_tokens = torch.full(
                        (padding_length,), 
                        pad_token_id, 
                        dtype=torch.long,  # 确保是long类型
                        device=new_input_ids[i].device
                    )
                    padded_input_id = torch.cat([pad_tokens, new_input_ids[i]])
                    
                    pad_attention = torch.zeros(
                        padding_length, 
                        dtype=torch.long,  # 确保是long类型
                        device=new_attention_masks[i].device
                    )
                    padded_attention_mask = torch.cat([pad_attention, new_attention_masks[i]])
                else:
                    padded_input_id = new_input_ids[i]
                    padded_attention_mask = new_attention_masks[i]
                
                padded_input_ids.append(padded_input_id)
                padded_attention_masks.append(padded_attention_mask)
            
            # 更新inputs - 确保数据类型正确
            inputs['input_ids'] = torch.stack(padded_input_ids, dim=0).long()
            inputs['attention_mask'] = torch.stack(padded_attention_masks, dim=0).long()
            
            # 如果有图像相关的输入，确保它们是正确的浮点类型
            for key in inputs:
                if key in ['pixel_values', 'image_embeds'] and isinstance(inputs[key], torch.Tensor):
                    if inputs[key].dtype.is_floating_point:
                        # 让forward方法处理类型转换
                        pass
            
        except Exception as e:
            print(f"Error during processor call: {e}")
            print(f"Type of batch_messages: {type(batch_messages)}")
            if batch_messages: 
                print(f"Type of first message: {type(batch_messages[0])}")
                print(f"First message content: {batch_messages[0]}")
            print(f"Batch assistant prefixes: {batch_assistant_prefixes}")
            raise
        
        return inputs
    def forward(self, image_or_video=None, labels=None, **args):
        # image_or_video_batch is the direct output from DataLoader (list of data_items)
        if image_or_video is None:
            raise ValueError("image_or_video_batch must be provided.")

        batched_inputs = self.processor_batch(image_or_video)
        
        # 将输入移到正确的设备并确保数据类型匹配
        device = self.dev
        model_dtype = next(self.model.parameters()).dtype
        
        for k, v in batched_inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                # 对于浮点数张量，转换为模型的数据类型
                if v.dtype.is_floating_point and k in ['pixel_values', 'image_embeds']:
                    v = v.to(model_dtype)
                # 对于整数张量（如input_ids, attention_mask），保持原有类型
                elif not v.dtype.is_floating_point:
                    v = v.long()  # 确保是long类型
                batched_inputs[k] = v
        
        # 如果有labels，也需要处理
        if labels is not None:
            labels = labels.to(device)
            if labels.dtype.is_floating_point:
                labels = labels.to(model_dtype)
            outputs = self.model(**batched_inputs, labels=labels)
        else:
            outputs = self.model(**batched_inputs)
    
        return outputs

    @property
    def dev(self):
        return next(self.model.parameters()).device


# Dataset classes remain the same as in the original code
class video_dataset(torch.utils.data.Dataset):
    def __init__(self, anno_file, data_prefix, phase, sample_types):
        super().__init__()
        self.video_infos = []
        self.phase = phase
        self.sample_types = sample_types 
        self.num_frames_to_sample = self.sample_types.get("clip_len", 16)

        with open(anno_file, "r") as fin:
            for line in fin:
                try:
                    line_split = line.strip().split(",")
                    if len(line_split) == 2:
                        filename_short, label_str = line_split
                        label = float(label_str)
                    elif len(line_split) == 4:
                        filename_short, a_str, t_str, label_str = line_split
                        label = float(label_str) 
                    else:
                        print(f"Skipping malformed line: {line.strip()}")
                        continue
                    
                    full_path = os.path.join(data_prefix, filename_short)
                    if not os.path.exists(full_path):
                        print(f"Warning: Video file not found {full_path}, skipping.")
                        continue
                    self.video_infos.append(dict(filename=full_path, label=label))
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
                    continue

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        video_info_entry = self.video_infos[idx]
        video_path = video_info_entry["filename"]
        video_label = video_info_entry["label"]

        video_frames = encode_video(video_path, num_frames=self.num_frames_to_sample)

        if not video_frames:
            print(f"Warning: Could not decode video: {video_path}. Using dummy frame.")
            dummy_frame = Image.new('RGB', (224, 224), color='black')
            video_frames = [dummy_frame]
        
        mock_video_inf = {"path": video_path, "frames_sampled": len(video_frames)}
        processed_video_data = {"info": mock_video_inf, "data": video_frames} 
        return processed_video_data, video_label


class ImageJsonDataset(torch.utils.data.Dataset):
    def __init__(self, dir, anno_file):
        self.dir = dir
        self.data = []
        try:
            with open(anno_file, 'r') as f:
                loaded_data = json.load(f)
                if "files" in loaded_data and isinstance(loaded_data["files"], list):
                    for item in loaded_data["files"]:
                        if 'image' in item and 'score' in item:
                            full_path = os.path.join(self.dir, item['image'])
                            if not os.path.exists(full_path):
                                print(f"Warning: Image file not found {full_path}, skipping.")
                                continue
                            self.data.append(item)
                else:
                    if isinstance(loaded_data, list):
                         for item in loaded_data:
                            if 'image' in item and 'score' in item:
                                full_path = os.path.join(self.dir, item['image'])
                                if not os.path.exists(full_path):
                                    print(f"Warning: Image file not found {full_path}, skipping.")
                                    continue
                                self.data.append(item)
        except Exception as e:
            print(f"Error loading JSON data: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.dir, item['image'])
        label = float(item['score'])

        image = Image.open(img_path).convert('RGB')

        width, height = image.size
        file_format = os.path.basename(img_path).split(".")[-1].upper()
        image_inf = {
            "Format": image.format if image.format else file_format,
            "File Size": f"{os.path.getsize(img_path)>>10:.0f}KB",
            "Resolution": f"{width}x{height}",
        }
        img_data_dict = {"info": image_inf, "data": image}
        return img_data_dict, label


class ImageCsvDataset(torch.utils.data.Dataset):
    def __init__(self, dir, anno_file, image_key, score_key):
        super().__init__()
        self.dir = dir
        self.data_list = []
        try:
            df = pd.read_csv(anno_file)
            if image_key in df.columns and score_key in df.columns:
                for _, row in df.iterrows():
                    full_path = os.path.join(self.dir, str(row[image_key]))
                    if not os.path.exists(full_path):
                        print(f"Warning: Image file not found {full_path}, skipping.")
                        continue
                    self.data_list.append((str(row[image_key]), float(row[score_key])))
        except Exception as e:
            print(f"Error reading CSV: {e}")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_name, label = self.data_list[idx]
        img_path = os.path.join(self.dir, img_name)

        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        file_format = os.path.basename(img_path).split(".")[-1].upper()
        image_inf = {
            "name": os.path.basename(img_path),
            "Format": image.format if image.format else file_format,
            "File Size": f"{os.path.getsize(img_path)>>10:.0f}KB",
            "Resolution": f"{width}x{height}",
        }
        img_data_dict = {"info": image_inf, "data": image}
        return img_data_dict, label


def uniform_sample(l, n, randomize=True):
    if not l: return []
    if n <= 0: return []
    if n >= len(l): return list(l)

    gap = len(l) / n
    if randomize:
        idxs = [int(i * gap + random.uniform(0, gap)) for i in range(n)]
        idxs = [min(i, len(l) - 1) for i in idxs]
    else:
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        idxs = [min(i, len(l) - 1) for i in idxs]
    return [l[i] for i in idxs]


def encode_video(video_path, num_frames=16, random_sampling=True):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        if len(vr) == 0:
            return []
            
        base_fps_divisor = 6
        avg_fps = vr.get_avg_fps()

        if avg_fps <= 0: 
            sample_fps = 1 
        elif random_sampling:
            sample_fps_divisor = max(0.1, base_fps_divisor + random.uniform(-1, 1))
            sample_fps = max(1, round(avg_fps / sample_fps_divisor))
        else:
            sample_fps = max(1, round(avg_fps / base_fps_divisor))
        
        frame_idx = [i for i in range(0, len(vr), sample_fps)]

        if not frame_idx: 
            frame_idx = [0] if len(vr) > 0 else []
                
        if len(frame_idx) > num_frames:
            frame_idx = uniform_sample(frame_idx, num_frames, randomize=random_sampling)
        
        if not frame_idx:
             return []

        frames_np = vr.get_batch(frame_idx).asnumpy()
        frames_pil = [Image.fromarray(v.astype('uint8')) for v in frames_np]
        return frames_pil
    except Exception as e:
        print(f"Error decoding video {video_path}: {e}")
        return []


def evaluate_iqa_internvl(model, val_dataset, val_embed, q_bench_good_poor_tokens, q_align_tokens, bsz=2):
    model.eval()
    num_workers = min(4, (os.cpu_count() // 2) if os.cpu_count() else 1)
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=num_workers, collate_fn=list_collate)
    
    val_pred_scores_topk, qbench_val_scores, qalign_val_scores, val_gt_scores = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc="InternVL3 Validation", ncols=100, disable=None)):
            image_or_video_data_batch, labels = batch[0], batch[1]
            if not image_or_video_data_batch: continue

            outputs = model(image_or_video_batch=image_or_video_data_batch)
            logits = outputs.logits.to(torch.float32)

            last_token_logits = logits[:, -1, :] 
            
            # Top-k Embeddings Method
            k = 100
            topk_results = torch.topk(last_token_logits, k, dim=-1)
            top_k_logits, top_k_indices = topk_results.values, topk_results.indices

            # 获取嵌入层 - 注意InternVL3的模型结构
            if hasattr(model.model, 'language_model'):
                embedding_layer = model.model.language_model.get_input_embeddings()
            else:
                embedding_layer = model.model.get_input_embeddings()
            
            top_k_embeddings = embedding_layer(top_k_indices.to(embedding_layer.weight.device))
            
            val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)
            weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
            
            score_topk = torch.sum(top_k_logits.to(weights.device) * weights, dim=-1)
            val_pred_scores_topk.extend(score_topk.cpu().tolist())
            
            # Q-Bench Method (good vs poor)
            if q_bench_good_poor_tokens:
                good_token_id, poor_token_id = q_bench_good_poor_tokens
                q_bench_token_ids = torch.tensor([good_token_id, poor_token_id], device=last_token_logits.device).long()
                binary_logits = last_token_logits[:, q_bench_token_ids]
                binary_probability = torch.softmax(binary_logits, dim=-1) 
                q_bench_score = binary_probability[:, 0]  # probability of "good"
                qbench_val_scores.extend(q_bench_score.cpu().tolist())

            # Q-Align Method (5-level rating)
            if q_align_tokens:
                q_align_token_ids = torch.tensor(q_align_tokens, device=last_token_logits.device).long()
                align_logits = last_token_logits[:, q_align_token_ids]
                align_probability = torch.softmax(align_logits, dim=-1)
                
                # 计算加权平均分数 (bad=1, poor=2, fair=3, good=4, excellent=5)
                rating_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=align_probability.device)
                q_align_score = torch.sum(align_probability * rating_weights, dim=-1)
                qalign_val_scores.extend(q_align_score.cpu().tolist())

            val_gt_scores.extend(labels.cpu().tolist())
        
    results_dict = {}
    val_gt_scores_tensor = torch.tensor(val_gt_scores, dtype=torch.float32)

    # Calculate for Top-k method
    if val_pred_scores_topk:
        val_pred_scores_topk_tensor = torch.tensor(val_pred_scores_topk, dtype=torch.float32)
        valid_indices_topk = ~torch.isnan(val_pred_scores_topk_tensor) & ~torch.isinf(val_pred_scores_topk_tensor)
        if not torch.all(valid_indices_topk):
            print(f"Warning (Top-k): Found {torch.sum(~valid_indices_topk)} NaN/Inf predicted scores. Excluding them.")
            val_pred_scores_topk_tensor = val_pred_scores_topk_tensor[valid_indices_topk]
            current_gt_scores_tensor = val_gt_scores_tensor[valid_indices_topk]
        else:
            current_gt_scores_tensor = val_gt_scores_tensor

        if len(val_pred_scores_topk_tensor) >= 2:
            srcc_topk, _ = spearmanr(val_pred_scores_topk_tensor.numpy(), current_gt_scores_tensor.numpy())
            plcc_topk, _ = pearsonr(val_pred_scores_topk_tensor.numpy(), current_gt_scores_tensor.numpy())
            results_dict["srcc_topk"] = srcc_topk
            results_dict["plcc_topk"] = plcc_topk
        else:
            results_dict["srcc_topk"] = 0.0
            results_dict["plcc_topk"] = 0.0
    else:
        results_dict["srcc_topk"] = 0.0
        results_dict["plcc_topk"] = 0.0

    # Calculate for Q-Bench method
    if qbench_val_scores:
        qbench_val_scores_tensor = torch.tensor(qbench_val_scores, dtype=torch.float32)
        valid_indices_qbench = ~torch.isnan(qbench_val_scores_tensor) & ~torch.isinf(qbench_val_scores_tensor)
        if not torch.all(valid_indices_qbench):
            print(f"Warning (Q-Bench): Found {torch.sum(~valid_indices_qbench)} NaN/Inf predicted scores. Excluding them.")
            qbench_val_scores_tensor = qbench_val_scores_tensor[valid_indices_qbench]
            current_gt_scores_tensor_q = val_gt_scores_tensor[valid_indices_qbench]
        else:
            current_gt_scores_tensor_q = val_gt_scores_tensor
            
        if len(qbench_val_scores_tensor) >= 2:
            srcc_qbench, _ = spearmanr(qbench_val_scores_tensor.numpy(), current_gt_scores_tensor_q.numpy())
            plcc_qbench, _ = pearsonr(qbench_val_scores_tensor.numpy(), current_gt_scores_tensor_q.numpy())
            results_dict["srcc_qbench"] = srcc_qbench
            results_dict["plcc_qbench"] = plcc_qbench
        else:
            results_dict["srcc_qbench"] = 0.0
            results_dict["plcc_qbench"] = 0.0
    else:
        results_dict["srcc_qbench"] = 0.0
        results_dict["plcc_qbench"] = 0.0

    # Calculate for Q-Align method
    if qalign_val_scores:
        qalign_val_scores_tensor = torch.tensor(qalign_val_scores, dtype=torch.float32)
        valid_indices_qalign = ~torch.isnan(qalign_val_scores_tensor) & ~torch.isinf(qalign_val_scores_tensor)
        if not torch.all(valid_indices_qalign):
            print(f"Warning (Q-Align): Found {torch.sum(~valid_indices_qalign)} NaN/Inf predicted scores. Excluding them.")
            qalign_val_scores_tensor = qalign_val_scores_tensor[valid_indices_qalign]
            current_gt_scores_tensor_qa = val_gt_scores_tensor[valid_indices_qalign]
        else:
            current_gt_scores_tensor_qa = val_gt_scores_tensor
            
        if len(qalign_val_scores_tensor) >= 2:
            srcc_qalign, _ = spearmanr(qalign_val_scores_tensor.numpy(), current_gt_scores_tensor_qa.numpy())
            plcc_qalign, _ = pearsonr(qalign_val_scores_tensor.numpy(), current_gt_scores_tensor_qa.numpy())
            results_dict["srcc_qalign"] = srcc_qalign
            results_dict["plcc_qalign"] = plcc_qalign
        else:
            results_dict["srcc_qalign"] = 0.0
            results_dict["plcc_qalign"] = 0.0
    else:
        results_dict["srcc_qalign"] = 0.0
        results_dict["plcc_qalign"] = 0.0
        
    return results_dict


if __name__ == "__main__":
    TASK = "IQA"

    INTERNVL_MODEL_PATH = "/media/ippl/LEXAR/Qwen/InternVL3-8B-hf/"
    LOCAL_FILES_ONLY = False

    YML_FILE_NAME_MAP = { "IQA": "iqa.yml", "IAA": "iaa.yml" }
    YML_FILE_NAME = YML_FILE_NAME_MAP[TASK]

    # Load dataset configuration
    try:
        with open(YML_FILE_NAME, "r") as f:
            opt = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: YAML configuration file '{YML_FILE_NAME}' not found. Exiting.")
        exit()
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML file '{YML_FILE_NAME}': {e}. Exiting.")
        exit()

    val_datasets_dict = {} 
    if opt and "test" in opt and isinstance(opt["test"], dict):
        for name, data_args_config in opt["test"].items():
            if isinstance(data_args_config, dict) and "type" in data_args_config and "args" in data_args_config:
                dataset_class_name = data_args_config["type"]
                dataset_args = data_args_config["args"]
                dataset_class = globals().get(dataset_class_name)
                if dataset_class:
                    try:
                        val_datasets_dict[name] = dataset_class(**dataset_args)
                        print(f"Loaded test dataset '{name}' with {len(val_datasets_dict[name])} samples.")
                    except Exception as e:
                        print(f"Error initializing dataset class {dataset_class_name} for '{name}': {e}")
                else:
                    print(f"Warning: Dataset class '{dataset_class_name}' for '{name}' not found.")
    else:
        print(f"Warning: 'test' section not found in '{YML_FILE_NAME}'.")

    if not val_datasets_dict:
        print("No valid test datasets loaded. Exiting.")
        exit()

    print("Loading InternVL3 model...")
    model = MultimodalQualityEvaluator(TASK, model_path=INTERNVL_MODEL_PATH, local_files_only=LOCAL_FILES_ONLY)
    print("InternVL3 Model loaded.")
    print(f"Model is on device: {model.dev}")

    text_dict_for_task = {   
        "IQA" : {
            "goodtext" : " perfect superb outstanding excellent fantastic stunning phenomenal brilliant magnificent amazing remarkable beautiful awesome breathtaking great good decent fine sharp clear suitable vibrant rich vivid bright colorful",
            "badtext" : " bad terrible awful poor horrible disappointing unacceptable inadequate deficient blurry fuzzy compromised chaotic distorted weak mediocre sub lacking unclear dark noisy low problematic insufficient"
        },
        "IAA" : {
            "goodtext": " beautiful stunning enchanting harmonious artistic pleasing exquisite elegant graceful balanced vibrant evocative poignant serene sublime picturesque appealing striking gorgeous charming delightful sophisticated",
            "badtext": " mediocre poorly dull bland chaotic displeasing lacking amateur overly subdued monotonous average cluttered uninspired unpleasant discordant garish mundane tacky glaring simplistic flat"
        }
    }
    
    current_text_map = text_dict_for_task[TASK]
    
    model.eval()
    computation_device = model.dev

    # with torch.no_grad():
    #     # 获取嵌入层
    #     if hasattr(model.model, 'language_model'):
    #         embedding_layer = model.model.language_model.get_input_embeddings()
    #     else:
    #         embedding_layer = model.model.get_input_embeddings()
            
    #     word_embeddings_dict = {}
    #     for category_name, words_string in current_text_map.items():
    #         inputs = model.tokenizer(words_string, return_tensors="pt", padding=True, truncation=True).to(computation_device)
    #         token_embeddings = embedding_layer(inputs["input_ids"])
    #         averaged_embedding = token_embeddings.mean(dim=1).squeeze(0) 
    #         word_embeddings_dict[category_name] = averaged_embedding

    # if word_embeddings_dict.get("goodtext") is not None and word_embeddings_dict.get("badtext") is not None:
    #     val_embed_for_scoring = (word_embeddings_dict["goodtext"] - word_embeddings_dict["badtext"]).to(torch.float32)
    #     print(f"val_embed_for_scoring computed. Shape: {val_embed_for_scoring.shape}, Device: {val_embed_for_scoring.device}")
    # else:
    #     raise RuntimeError("Could not compute good/bad text embeddings for val_embed_for_scoring.")

    # # For Q-Bench style evaluation
    # try:
    #     good_tokens = model.tokenizer.encode("good", add_special_tokens=False)
    #     poor_tokens = model.tokenizer.encode("poor", add_special_tokens=False)
    #     if good_tokens and poor_tokens:
    #         q_bench_good_poor_token_ids = (good_tokens[0], poor_tokens[0])
    #         print(f"Q-Bench 'good' token ID: {q_bench_good_poor_token_ids[0]}, 'poor' token ID: {q_bench_good_poor_token_ids[1]}")
    #     else:
    #         q_bench_good_poor_token_ids = None
    #         print("Failed to get token IDs for 'good'/'poor'. Q-Bench part will be skipped.")
    # except:
    #     q_bench_good_poor_token_ids = None
    #     print("Failed to get token IDs for 'good'/'poor'. Q-Bench part will be skipped.")

    # # For Q-Align style evaluation (5-level rating)
    # try:
    #     q_align_words = ["bad", "poor", "fair", "good", "excellent"]
    #     q_align_token_ids = []
    #     for word in q_align_words:
    #         tokens = model.tokenizer.encode(word, add_special_tokens=False)
    #         if tokens:
    #             q_align_token_ids.append(tokens[0])
    #         else:
    #             print(f"Warning: Could not tokenize '{word}' for Q-Align.")
    #             q_align_token_ids = None
    #             break
        
    #     if q_align_token_ids and len(q_align_token_ids) == 5:
    #         print(f"Q-Align token IDs: {dict(zip(q_align_words, q_align_token_ids))}")
    #     else:
    #         q_align_token_ids = None
    #         print("Failed to get all token IDs for Q-Align. Q-Align part will be skipped.")
    # except:
    #     q_align_token_ids = None
    #     print("Failed to get token IDs for Q-Align. Q-Align part will be skipped.")

    # batch_size = 1
    # for name, current_val_dataset in val_datasets_dict.items():
    #     print(f"\nEvaluating on dataset: {name} (Size: {len(current_val_dataset)})")
    #     if len(current_val_dataset) == 0:
    #         print(f"Dataset {name} is empty. Skipping.")
    #         continue
            
    #     results = evaluate_iqa_internvl(
    #         model, current_val_dataset, val_embed_for_scoring, 
    #         q_bench_good_poor_tokens=q_bench_good_poor_token_ids,
    #         q_align_tokens=q_align_token_ids,
    #         bsz=batch_size, 
    #     )
    #     print(f"******** Results for {name}: *********")
    #     for key, value in results.items():
    #         if isinstance(value, float):
    #             print(f"  {key}:\t {value:.4f}")
    #         else:
    #             print(f"  {key}:\t {value}")

    # print("\nEvaluation finished.")

    from owl3_zeroshot import get_top_logits

    get_top_logits("/home/ippl/xxr/mPLUG-Owl/mPLUG-Owl3/assets/testimg", model, batch_size=1, topk=10)