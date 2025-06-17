import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from PIL import Image
import os
import yaml
import tqdm
import time
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import pandas as pd
import json
import random
# import ffmpeg # Not directly used here, but decord uses it indirectly
from decord import VideoReader, cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Dataset Collate Function (as per user's setup) ---
def list_collate(batch):
    data_items = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return data_items, torch.tensor(labels, dtype=torch.float32)

# --- Dataset Classes (adapted from user's owl3_zeroshot.py) ---
class video_dataset(torch.utils.data.Dataset):
    def __init__(self, anno_file, data_prefix, phase, sample_types):
        super().__init__()
        self.video_infos = []
        self.phase = phase
        self.sample_types = sample_types 
        self.num_frames_to_sample = self.sample_types.get("clip_len", 8) # LLaVA examples use 8 or 32

        with open(anno_file, "r") as fin:
            for line in fin:
                try:
                    line_split = line.strip().split(",")
                    if len(line_split) == 4: # filename, a, t, label
                        filename, _, _, label_str = line_split
                    elif len(line_split) == 2: # filename, label (simplified)
                        filename, label_str = line_split
                    else:
                        print(f"Skipping malformed line in {anno_file}: {line.strip()}")
                        continue
                    label = float(label_str)
                    full_path = os.path.join(data_prefix, filename)
                    if not os.path.exists(full_path):
                        print(f"Warning: Video file not found {full_path}, skipping.")
                        continue
                    self.video_infos.append(dict(filename=full_path, label=label))
                except ValueError:
                    print(f"Skipping line due to ValueError (e.g. non-float label): {line.strip()}")
                    continue


    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        video_info_entry = self.video_infos[idx]
        video_path = video_info_entry["filename"]
        video_label = video_info_entry["label"]

        video_frames = encode_video(video_path, num_frames=self.num_frames_to_sample)

        if not video_frames:
            print(f"Warning: Could not decode video or video is empty: {video_path}. Using a dummy frame.")
            # Create a single black dummy frame if encoding fails
            dummy_frame = Image.new('RGB', (224, 224), color='black')
            video_frames = [dummy_frame]
        
        # Mock video_inf for consistency with user's data structure
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
                            print(f"Skipping item in {anno_file} due to missing 'image' or 'score': {item}")
                else:
                    print(f"Warning: 'files' key not found or not a list in {anno_file}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {anno_file}")
        except FileNotFoundError:
            print(f"Annotation file not found: {anno_file}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.dir, item['image'])
        label = item['score']
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file {img_path} not found during getitem. Returning dummy.")
            image = Image.new('RGB', (224, 224), color='black') # Dummy image
            label = 0.0 # Dummy label
        except Exception as e:
            print(f"Error opening image {img_path}: {e}. Returning dummy.")
            image = Image.new('RGB', (224, 224), color='black')
            label = 0.0
        
        mock_image_inf = {"path": img_path, "resolution": f"{image.width}x{image.height}"}
        img_data_dict = {"info": mock_image_inf, "data": image}
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
            else:
                print(f"Warning: '{image_key}' or '{score_key}' not in columns of {anno_file}")
        except FileNotFoundError:
            print(f"Annotation file not found: {anno_file}")
        except Exception as e:
            print(f"Error reading CSV {anno_file}: {e}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_name, label = self.data_list[idx]
        img_path = os.path.join(self.dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file {img_path} not found during getitem. Returning dummy.")
            image = Image.new('RGB', (224, 224), color='black')
            label = 0.0
        except Exception as e:
            print(f"Error opening image {img_path}: {e}. Returning dummy.")
            image = Image.new('RGB', (224, 224), color='black')
            label = 0.0
        
        mock_image_inf = {"path": img_path, "resolution": f"{image.width}x{image.height}"}
        img_data_dict = {"info": mock_image_inf, "data": image}
        return img_data_dict, label

def encode_video(video_path, num_frames=8):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames_in_video = len(vr)
        if total_frames_in_video == 0:
            # print(f"Warning: Video {video_path} has 0 frames.")
            return []

        if total_frames_in_video <= num_frames:
            frame_idx = list(range(total_frames_in_video))
        else:
            # Uniformly sample num_frames indices
            # np.linspace(0, total_frames_in_video - 1, num_frames).astype(int)
            frame_idx = [int(i * (total_frames_in_video - 1) / (num_frames - 1)) if num_frames > 1 else 0 for i in range(num_frames)]
            frame_idx = sorted(list(set(frame_idx))) # Ensure unique, sorted indices

        if not frame_idx and total_frames_in_video > 0:
            frame_idx = [0] 
        if not frame_idx:
            return []

        frames_data = vr.get_batch(frame_idx).asnumpy()
        frames_pil = [Image.fromarray(frame.astype('uint8')) for frame in frames_data]
        return frames_pil
    except RuntimeError as e: # decord can raise RuntimeError for corrupt videos
        print(f"RuntimeError decoding video {video_path}: {e}")
        return []
    except Exception as e:
        print(f"General error decoding video {video_path}: {e}")
        return []
# --- End Dataset Classes ---

class MultimodalQualityEvaluatorLLaVANext(nn.Module):
    def __init__(
        self,
        task="IQA",
        model_path="llava-hf/LLaVA-NeXT-Video-7B-hf",
        local_files_only=True, # Set to True if model is downloaded and you want to force local
    ):
        super().__init__()
        self.task = task
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else torch.bfloat16
        if not torch.cuda.is_available(): # CPU
            dtype = torch.float32 
            print("Warning: CUDA not available. Using CPU with torch.float32 for LLaVA-NeXT model.")
        elif dtype == torch.bfloat16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            print("Warning: bfloat16 specified but not supported. Falling back to float16.")
            dtype = torch.float16
        
        print(f"LLaVA-NeXT model will be loaded with dtype: {dtype}")

        self.LLM = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            # low_cpu_mem_usage=True,
            attn_implementation="sdpa", 
            local_files_only=local_files_only,
            device_map="auto" if torch.cuda.is_available() else None, # Use auto for CUDA, None for CPU
        )

        self.LLMprocessor = LlavaNextVideoProcessor.from_pretrained(model_path, local_files_only=local_files_only, patch_size=14, use_fast=True)
        self.tokenizer = self.LLMprocessor.tokenizer

    def _prepare_prompt_and_media(self, data_item):
        media_data = data_item["data"]
        is_video = isinstance(media_data, list) and all(isinstance(frame, Image.Image) for frame in media_data)
        
        media_type_placeholder = "video" if is_video else "image"
        media_type_str = "video" if is_video else "image"

        if self.task == "IQA":
            user_prompt_text = f"Taking into account the details and the rationality of the {media_type_str}, how would you rate the quality of this {media_type_str}?"
        elif self.task == "IAA":
            user_prompt_text = f"Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this {media_type_str}?"
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Structured content for the user's turn.
        # LLaVA README example: [{"type": "text", "text": "Why is this video funny?"}, {"type": "video"}]
        # Let's try media placeholder first, then text, as <image> usually precedes query.
        structured_user_content = [
            {"type": media_type_placeholder}, 
            {"type": "text", "text": user_prompt_text}
        ]
        
        conversation_for_sample = [{"role": "user", "content": structured_user_content}]
        

        prompt_base = self.LLMprocessor.apply_chat_template(
            conversation_for_sample,
            tokenize=False,
            add_generation_prompt=True
        )

        final_prompt = prompt_base + f"The quality of the {media_type_str} is quite"
        
        current_image = media_data if not is_video else None
        current_video = media_data if is_video else None
        
        return final_prompt, current_image, current_video

    def processor(self, data_and_info_batch):
        batch_prompts, batch_images, batch_videos = [], [], []
        has_any_image, has_any_video = False, False

        for data_item in data_and_info_batch:
            prompt, image, video = self._prepare_prompt_and_media(data_item)
            batch_prompts.append(prompt)
            batch_images.append(image)
            batch_videos.append(video)
            if image is not None: has_any_image = True
            if video is not None: has_any_video = True

        images_arg = batch_images if has_any_image else None
        videos_arg = batch_videos if has_any_video else None
        
        inputs = self.LLMprocessor(
            text=batch_prompts, images=images_arg, videos=videos_arg,
            padding=True, return_tensors="pt"
        )
        return inputs

    def forward(self, image_or_video_batch=None, **args):
        if image_or_video_batch is None:
            raise ValueError("image_or_video_batch must be provided.")

        batched_inputs = self.processor(image_or_video_batch)
        # Move tensors to the model's device
        device = self.dev
        for k, v in batched_inputs.items():
            if isinstance(v, torch.Tensor):
                batched_inputs[k] = v.to(device)
            elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v): # For potential list of tensors
                 batched_inputs[k] = [i.to(device) for i in v]
        
        outputs = self.LLM(**batched_inputs)
        return outputs

    @property
    def dev(self):
        return next(self.LLM.parameters()).device

def evaluate_iqa_llava(model, val_dataset, val_embed, q_bench_good_poor_tokens, bsz=8, device_str="cuda:0"):
    model.eval()
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1), collate_fn=list_collate)
    
    val_pred_scores_topk, qbench_val_scores, val_gt_scores = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc="LLaVA-NeXT Validation", ncols=100, disable=None)):
            image_or_video_data_batch, labels = batch[0], batch[1]
            if not image_or_video_data_batch: continue # Skip if batch is empty

            outputs = model(image_or_video_batch=image_or_video_data_batch)
            logits = outputs.logits.to(torch.float32) 

            last_token_logits = logits[:, -1, :]
            
            # --- Top-k Embeddings Method ---
            k = 300
            topk_results = torch.topk(last_token_logits, k, dim=-1)
            top_k_logits, top_k_indices = topk_results.values, topk_results.indices

            embedding_layer = model.LLM.get_input_embeddings()
            top_k_embeddings = embedding_layer(top_k_indices.to(embedding_layer.weight.device))
            
            val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)
            weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
            
            score_topk = torch.sum(top_k_logits.to(weights.device) * weights, dim=-1)
            val_pred_scores_topk.extend(score_topk.cpu().tolist())
            
            # --- Q-Bench Method (good/poor tokens) ---
            if q_bench_good_poor_tokens:
                good_token_id, poor_token_id = q_bench_good_poor_tokens
                # Ensure selected_tokens are valid indices for last_token_logits
                # Create a tensor of these specific token IDs for gathering
                # Check if token IDs are within the vocab size
                vocab_size = last_token_logits.shape[-1]
                if good_token_id < vocab_size and poor_token_id < vocab_size:
                    q_bench_token_ids = torch.tensor([good_token_id, poor_token_id], device=last_token_logits.device).long()
                    binary_logits = last_token_logits[:, q_bench_token_ids] # Shape: (batch_size, 2)
                    binary_probability = torch.softmax(binary_logits, dim=-1) 
                    q_bench_score = binary_probability[:, 0] # Probability of "good"
                    qbench_val_scores.extend(q_bench_score.cpu().tolist())
                else:
                    # Append a default/neutral score or handle as error if token IDs are out of bounds
                    # For simplicity, appending 0.5 (neutral) if IDs are bad, or you could skip.
                    print(f"Warning: Q-Bench token IDs ({good_token_id}, {poor_token_id}) out of vocab size ({vocab_size}). Appending neutral score.")
                    qbench_val_scores.extend([0.5] * last_token_logits.shape[0])


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
            current_gt_scores_tensor_topk = val_gt_scores_tensor[valid_indices_topk]
        else:
            current_gt_scores_tensor_topk = val_gt_scores_tensor

        if len(val_pred_scores_topk_tensor) >= 2 and len(current_gt_scores_tensor_topk) == len(val_pred_scores_topk_tensor):
            srcc_topk, _ = spearmanr(val_pred_scores_topk_tensor.numpy(), current_gt_scores_tensor_topk.numpy())
            plcc_topk, _ = pearsonr(val_pred_scores_topk_tensor.numpy(), current_gt_scores_tensor_topk.numpy())
            results_dict["srcc_topk"] = srcc_topk
            results_dict["plcc_topk"] = plcc_topk
        else:
            results_dict["srcc_topk"] = 0.0
            results_dict["plcc_topk"] = 0.0
    else:
        results_dict["srcc_topk"] = 0.0
        results_dict["plcc_topk"] = 0.0

    # Calculate for Q-Bench method
    if qbench_val_scores: # Ensure qbench_val_scores were populated
        qbench_val_scores_tensor = torch.tensor(qbench_val_scores, dtype=torch.float32)
        # Ensure qbench_val_scores has same length as gt_scores before filtering
        if len(qbench_val_scores_tensor) == len(val_gt_scores_tensor):
            valid_indices_qbench = ~torch.isnan(qbench_val_scores_tensor) & ~torch.isinf(qbench_val_scores_tensor)
            if not torch.all(valid_indices_qbench):
                print(f"Warning (Q-Bench): Found {torch.sum(~valid_indices_qbench)} NaN/Inf predicted scores. Excluding them.")
                qbench_val_scores_tensor = qbench_val_scores_tensor[valid_indices_qbench]
                current_gt_scores_tensor_qbench = val_gt_scores_tensor[valid_indices_qbench]
            else:
                current_gt_scores_tensor_qbench = val_gt_scores_tensor
                
            if len(qbench_val_scores_tensor) >= 2 and len(current_gt_scores_tensor_qbench) == len(qbench_val_scores_tensor):
                srcc_qbench, _ = spearmanr(qbench_val_scores_tensor.numpy(), current_gt_scores_tensor_qbench.numpy())
                plcc_qbench, _ = pearsonr(qbench_val_scores_tensor.numpy(), current_gt_scores_tensor_qbench.numpy())
                results_dict["srcc_qbench"] = srcc_qbench
                results_dict["plcc_qbench"] = plcc_qbench
            else:
                results_dict["srcc_qbench"] = 0.0
                results_dict["plcc_qbench"] = 0.0
        else:
            print(f"Warning (Q-Bench): Mismatch in number of Q-Bench scores ({len(qbench_val_scores_tensor)}) and GT scores ({len(val_gt_scores_tensor)}). Skipping Q-Bench metrics.")
            results_dict["srcc_qbench"] = 0.0
            results_dict["plcc_qbench"] = 0.0
    else:
        results_dict["srcc_qbench"] = 0.0
        results_dict["plcc_qbench"] = 0.0
        
    return results_dict, val_pred_scores_topk, val_gt_scores # Return only topk preds for now, or adjust as needed

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    selected_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device_str)
    print(f"Using device: {device}")

    TASK = "IQA" 
    LLAVA_MODEL_PATH = "/media/ippl/LEXAR/Qwen/LLaVA-NeXT-Video-7B-hf"
    # Set to True if you have the model downloaded and want to ensure it's used,
    # or if you are offline.
    # LOCAL_FILES_ONLY = True # Example: if model is in ~/.cache/huggingface/hub
    LOCAL_FILES_ONLY = False # Default: try to download if not present

    YML_FILE_NAME = "iqa.yml" if TASK == "IQA" else "iaa.yml"
    
    # --- Create dummy data and config if not present ---
    sample_data_dir = "./sample_llava_data"
    sample_img_dir = os.path.join(sample_data_dir, "images")
    sample_vid_dir = os.path.join(sample_data_dir, "videos") # For video dataset example
    
    if not os.path.exists(YML_FILE_NAME):
        print(f"'{YML_FILE_NAME}' not found. Creating dummy config and data in '{sample_data_dir}'.")
        os.makedirs(sample_img_dir, exist_ok=True)
        # os.makedirs(sample_vid_dir, exist_ok=True) # If testing video

        try:
            Image.new('RGB', (224, 224), color = 'red').save(os.path.join(sample_img_dir, "dummy1.jpg"))
            Image.new('RGB', (224, 224), color = 'blue').save(os.path.join(sample_img_dir, "dummy2.jpg"))
        except Exception as e:
            print(f"Could not create dummy images: {e}")

        dummy_anno_content = {
            "files": [
                {"image": "dummy1.jpg", "score": 0.8},
                {"image": "dummy2.jpg", "score": 0.3}
            ]
        }
        sample_anno_json_path = os.path.join(sample_data_dir, "sample_anno.json")
        with open(sample_anno_json_path, "w") as f:
            json.dump(dummy_anno_content, f)
        
        dummy_yml_content = f"""
test:
  sample_image_dataset:
    type: ImageJsonDataset
    args:
      dir: "{sample_img_dir}" # Use absolute or correct relative path
      anno_file: "{sample_anno_json_path}"
#  sample_video_dataset: # Example for video
#    type: video_dataset
#    args:
#      data_prefix: "{sample_vid_dir}"
#      anno_file: "{os.path.join(sample_data_dir, "sample_video_anno.txt")}" # Create this file
#      phase: "test"
#      sample_types: {{"clip_len": 8}}
"""
        with open(YML_FILE_NAME, "w") as f:
            f.write(dummy_yml_content)
        print(f"Created dummy '{YML_FILE_NAME}' and sample data.")
        # if testing video, create dummy sample_video_anno.txt:
        # e.g. dummy_video.mp4,0.7 (and place a dummy_video.mp4 in sample_vid_dir)

    # --- Load dataset configuration ---
    try:
        with open(YML_FILE_NAME, "r") as f:
            opt = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: YAML configuration file '{YML_FILE_NAME}' not found even after attempting to create a dummy. Exiting.")
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
                        print(f"Loaded dataset '{name}' with {len(val_datasets_dict[name])} samples.")
                        if len(val_datasets_dict[name]) == 0:
                             print(f"Warning: Dataset '{name}' is empty. Check paths and annotation files in '{YML_FILE_NAME}'.")
                    except Exception as e:
                        print(f"Error initializing dataset class {dataset_class_name} for '{name}': {e}")
                else:
                    print(f"Warning: Dataset class '{dataset_class_name}' for '{name}' not found.")
            else:
                 print(f"Warning: Invalid configuration for dataset '{name}' in '{YML_FILE_NAME}'. Missing 'type' or 'args'.")
    else:
        print(f"Warning: 'test' section not found or invalid in '{YML_FILE_NAME}'. No datasets loaded.")

    if not val_datasets_dict:
        print("No valid test datasets loaded. Exiting.")
        exit()

    print("Loading LLaVA-NeXT-Video model...")
    llava_model = MultimodalQualityEvaluatorLLaVANext(TASK, model_path=LLAVA_MODEL_PATH, local_files_only=LOCAL_FILES_ONLY)
    # .to(device) is not strictly needed here if device_map="auto" is used and model is large.
    # For smaller models or single GPU, .to(device) is fine.
    # If device_map="auto", the model's parameters are already on the correct devices.
    # We will move inputs to model.dev in the forward pass.
    print(f"Model loaded. Primary parameter device: {llava_model.dev}")


    text_dict_for_task = {
        "IQA": {
            "goodtext": "perfect superb outstanding excellent fantastic stunning phenomenal brilliant magnificent amazing remarkable beautiful awesome breathtaking great good decent fine sharp clear suitable vibrant rich vivid bright colorful",
            "badtext": "bad terrible awful poor horrible disappointing unacceptable inadequate deficient blurry fuzzy compromised chaotic distorted weak mediocre sub lacking unclear dark noisy low problematic insufficient"
        },
        "IAA": {
            "goodtext": "beautiful stunning enchanting harmonious artistic pleasing exquisite elegant graceful balanced vibrant evocative poignant serene sublime picturesque appealing striking gorgeous charming delightful sophisticated",
            "badtext": "mediocre poorly dull bland chaotic displeasing lacking amateur overly subdued monotonous average cluttered uninspired unpleasant discordant garish mundane tacky glaring simplistic flat"
        }
    }
    
    current_text_map = text_dict_for_task[TASK]
    good_embedding_vector, bad_embedding_vector = None, None
    
    llava_model.eval()
    with torch.no_grad():
        embedding_layer = llava_model.LLM.get_input_embeddings()
        for word_category, text_string in current_text_map.items():
            inputs = llava_model.tokenizer(text_string, return_tensors="pt", padding=True, truncation=True).to(device)
            token_embeddings = embedding_layer(inputs["input_ids"]) 
            averaged_embedding = token_embeddings.mean(dim=1).squeeze(0)
            if word_category == "goodtext": good_embedding_vector = averaged_embedding
            elif word_category == "badtext": bad_embedding_vector = averaged_embedding

    if good_embedding_vector is not None and bad_embedding_vector is not None:
        val_embed_for_scoring = (good_embedding_vector - bad_embedding_vector).to(torch.float32)
        print(f"val_embed_for_scoring shape: {val_embed_for_scoring.shape}, device: {val_embed_for_scoring.device}")
    else:
        raise RuntimeError("Could not compute good/bad text embeddings for val_embed.")

    # For Q-Bench style evaluation with LLaVA
    q_bench_good_poor_token_ids = None
    try:
        # LLaVA tokenizer might not give single tokens for "good"/"poor".
        # This is a simple attempt; might need refinement based on LLaVA's vocab.
        good_token_id = llava_model.tokenizer.encode("good", add_special_tokens=False)
        poor_token_id = llava_model.tokenizer.encode("poor", add_special_tokens=False)

        if good_token_id and poor_token_id: # Check if encoding returned anything
            # Take the first token if multiple are returned (common for words not in vocab as single units)
            good_id = good_token_id[0]
            poor_id = poor_token_id[0]
            q_bench_good_poor_token_ids = (good_id, poor_id)
            print(f"LLaVA Q-Bench 'good' (first token) ID: {good_id}, 'poor' (first token) ID: {poor_id}")
        else:
            print("Warning: Could not encode 'good' or 'poor' for LLaVA Q-Bench. Q-Bench scores might be disabled or incorrect.")
            
    except Exception as e:
        print(f"Warning: Error encoding 'good'/'poor' for LLaVA Q-Bench: {e}. Q-Bench scores might be disabled.")


    batch_size = 1 # Start with 1 for LLaVA-7B, increase if memory allows
    for name, current_val_dataset in val_datasets_dict.items():
        print(f"\nEvaluating on dataset: {name} (Size: {len(current_val_dataset)})")
        if len(current_val_dataset) == 0:
            print(f"Dataset {name} is empty. Skipping.")
            continue
            
        results_dict, _, _ = evaluate_iqa_llava(
            llava_model, current_val_dataset, val_embed_for_scoring, 
            q_bench_good_poor_tokens=q_bench_good_poor_token_ids,
            bsz=batch_size, device_str=selected_device_str
        )
        print(f"Results for {name}:")
        if "srcc_topk" in results_dict and results_dict['srcc_topk'] is not None:
            print(f"  SRCC (Top-k): {results_dict['srcc_topk']:.4f}")
            print(f"  PLCC (Top-k): {results_dict['plcc_topk']:.4f}")
        else:
            print(f"  SRCC (Top-k): N/A")
            print(f"  PLCC (Top-k): N/A")

        if "srcc_qbench" in results_dict and results_dict['srcc_qbench'] is not None:
            print(f"  SRCC (Q-Bench): {results_dict['srcc_qbench']:.4f}")
            print(f"  PLCC (Q-Bench): {results_dict['plcc_qbench']:.4f}")
        else:
            print(f"  SRCC (Q-Bench): N/A")
            print(f"  PLCC (Q-Bench): N/A")

    print("\nEvaluation finished.")