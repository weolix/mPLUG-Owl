import os
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1" # 去除颜色编码
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["AV_LOG_LEVEL"] = "quiet"  
os.environ["FFMPEG_LOGLEVEL"] = "quiet" 

import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import dataset
from quality_plugowl3 import QualityOwl3Model
import yaml
import tqdm, time
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
date_str = time.strftime("%Y-%m-%d", time.localtime())
time_str = time.strftime("%H:%M:%S", time.localtime())
run_dir = f"runs/{date_str}/{time_str}"
writer = SummaryWriter(run_dir)


def trainq(model, val_datasets, optimizer, hylayers, gradient_accumulation_steps=4):
    epochs = 5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    global_step = 0
    best_plcc = 0.7
    for epoch in range(epochs):
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

       # 验证
        model.eval()
        for name, data in val_datasets["test"].items():
            valdataloader = DataLoader(data, batch_size=5, shuffle=True, num_workers=8, collate_fn=dataset.dict_simply_collate)
            
            with torch.no_grad():
                val_pred_scores = []
                val_gt_scores = []
                for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"{name} Validation", ncols=100)):
                    aesthetic = batch["aesthetic"]
                    technical = batch["technical"] 
                    labels = batch["gt_label"].to(model.LLM.device)

                    outputs = model(aesthetic=aesthetic, technical=technical, labels=labels)
                    plccloss, rankloss, score = outputs.loss
                    val_pred_scores += (score.cpu().tolist())
                    val_gt_scores += (labels.cpu().tolist())
                
                    # 清理不需要的变量
                    del batch, aesthetic, technical, labels, outputs, plccloss, rankloss, score
                    torch.cuda.empty_cache()  # 定期清理GPU缓存
                
                val_pred_scores = torch.tensor(val_pred_scores)
                val_gt_scores = torch.tensor(val_gt_scores)
                spearmanrcc = spearmanr(val_pred_scores, val_gt_scores)
                pearsonrcc = pearsonr(val_pred_scores, val_gt_scores)
                writer.add_scalar(f"Spearmanr/val-{name}", spearmanrcc[0], epoch)
                writer.add_scalar(f"Pearsonr/val-{name}", pearsonrcc[0], epoch)
                print(f"{name} eval Spearmanr: {spearmanrcc[0]:.4f}, Pearsonr: {pearsonrcc[0]:.4f}")

        # 训练
        model.train()
        total_loss = 0
        pred_scores = []
        gt_scores = []
        optimizer.zero_grad()  # 在循环开始前清零梯度
        
        for name, data in val_datasets["train"].items():
            dataloader = DataLoader(data, batch_size=5, shuffle=True, num_workers=8, collate_fn=dataset.dict_simply_collate)
            for i, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)):
                aesthetic = batch["aesthetic"]
                technical = batch["technical"] 
                labels = batch["gt_label"].to(model.LLM.device)

                outputs = model(aesthetic=aesthetic, technical=technical, labels=labels)
                plccloss, rankloss, score = outputs.loss
                loss = plccloss #+ rankloss
                
                # 梯度累积：除以累积步数来缩放损失
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # 记录loss和其他指标
                current_loss = loss.item() * gradient_accumulation_steps  # 还原为原始损失值以便记录
                total_loss += current_loss
                
                pred_scores.extend(score.detach().cpu().tolist())
                gt_scores.extend(labels.detach().cpu().tolist())
                
                # 每gradient_accumulation_steps步更新一次参数
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 记录到tensorboard
                    writer.add_scalar("Loss/step", current_loss, global_step)
                    global_step += 1

                # 清理不需要的变量
                del outputs, plccloss, rankloss, score, loss
                torch.cuda.empty_cache()  # 定期清理GPU缓存

            writer.add_scalar("Loss/epoch", total_loss, epoch)
            print(f"Epoch {epoch+1} Average Loss: {total_loss:.4f}")

            pred_scores = torch.tensor(pred_scores)
            gt_scores = torch.tensor(gt_scores)
            spearmanrcc, pearsonrcc = spearmanr(pred_scores, gt_scores), pearsonr(pred_scores, gt_scores)
            writer.add_scalar("Spearmanr/train", spearmanrcc[0], epoch)
            writer.add_scalar("Pearsonr/train", pearsonrcc[0], epoch)
            print(f"train Spearmanr: {spearmanrcc[0]:.4f}, Pearsonr: {pearsonrcc[0]:.4f}")

        try:
            # 创建一个字典来存储所有需要保存的参数, 训练完直接保存
            save_dict = {
                'q2t_state': model.quality2text_model.state_dict(),
                'attn_states': {}
            }
            
            # 保存训练的hyperlayer的v_kv_proj层的参数
            for layer_idx in hylayers:
                save_dict['attn_states'][f'layer_{layer_idx}'] = (
                    model.LLM.language_model.model.layers[layer_idx-1]
                    .self_attn.v_kv_proj.state_dict()
                )
            
            # 保存为单个文件
            torch.save(save_dict, f"{run_dir}/model_epoch_{epoch}.pth")
            print(f"model saved at epoch {epoch+1} in {run_dir}")
        except Exception as e:
            print(f"保存失败: {e}")

        # EMA (Exponential Moving Average) implementation
        # Create EMA parameters if it's not the first epoch
        if epoch > 0:
            try:
                # Load the previous epoch's model
                prev_checkpoint = torch.load(f"{run_dir}/model_epoch_{epoch-1}.pth")
                
                # Apply EMA to quality2text model parameters
                ema_decay = 0.9  # EMA decay factor (adjust as needed)
                current_q2t_state = model.quality2text_model.state_dict()
                prev_q2t_state = prev_checkpoint['q2t_state']
                
                for key in current_q2t_state:
                    if key in prev_q2t_state:
                        current_q2t_state[key] = ema_decay * prev_q2t_state[key] + (1 - ema_decay) * current_q2t_state[key]
                
                model.quality2text_model.load_state_dict(current_q2t_state)
                
                # Apply EMA to attention layers
                for layer_idx in hylayers:
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in prev_checkpoint['attn_states']:
                        current_attn_state = model.LLM.language_model.model.layers[layer_idx-1].self_attn.v_kv_proj.state_dict()
                        prev_attn_state = prev_checkpoint['attn_states'][layer_key]
                        
                        for key in current_attn_state:
                            if key in prev_attn_state:
                                current_attn_state[key] = ema_decay * prev_attn_state[key] + (1 - ema_decay) * current_attn_state[key]
                        
                        model.LLM.language_model.model.layers[layer_idx-1].self_attn.v_kv_proj.load_state_dict(current_attn_state)
                
                print(f"Applied EMA with decay {ema_decay} from epoch {epoch} weights")
            except Exception as e:
                print(f"Failed to apply EMA: {e}")

        
 

        scheduler.step()
    writer.close()


def evaluate(model, val_dataset):
    # 验证
    model.eval()

    valdataloader = DataLoader(val_dataset, batch_size=5, shuffle=True, num_workers=8, collate_fn=dataset.dict_simply_collate)
    
    with torch.no_grad():
        val_pred_scores = []
        val_gt_scores = []
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"Validation", ncols=100)):
            aesthetic = batch["aesthetic"]
            technical = batch["technical"] 
            labels = batch["gt_label"].to(model.LLM.device)

            outputs = model(aesthetic=aesthetic, technical=technical, labels=labels)
            plccloss, rankloss, score = outputs.loss
            val_pred_scores += (score.cpu().tolist())
            val_gt_scores += (labels.cpu().tolist())
        val_pred_scores = torch.tensor(val_pred_scores)
        val_gt_scores = torch.tensor(val_gt_scores)
        spearmanrcc = spearmanr(val_pred_scores, val_gt_scores)
        pearsonrcc = pearsonr(val_pred_scores, val_gt_scores)

        print(f"eval Spearmanr: {spearmanrcc[0]:.4f}, Pearsonr: {pearsonrcc[0]:.4f}")
        


def load_model(model, load_run, load_epoch, hylayers):

    checkpoint = torch.load(f"{load_run}/model_epoch_{load_epoch}.pth")
    
    # 加载quality2text模型参数
    model.quality2text_model.load_state_dict(checkpoint['q2t_state'])
    
    # 加载attention层参数
    for layer_idx in hylayers:
        model.LLM.language_model.model.layers[layer_idx-1].self_attn.v_kv_proj.load_state_dict(
            checkpoint['attn_states'][f'layer_{layer_idx}']
        )


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载配置文件，假设 data.yml 中包含训练参数
    opt = yaml.safe_load(open("data.yml", "r"))
    val_datasets = {}
    for phase, datasets in opt.items():
        val_datasets[phase] = {}
        for name, data_args in datasets.items():
            val_datasets[phase][name] = getattr(dataset, data_args["type"])(data_args)


    # 创建模型，启用 quality brance 进行训练
    hylayers = [
        # 7, 15, 23, 26, # original layers
        # 19, # add
        28, # add
    ]
    model = QualityOwl3Model(new_layers=hylayers).to(device)
    q2t_param = [{'params': model.quality2text_model.parameters()}]
    attn_param = [{'params': model.LLM.language_model.model.layers[k-1].self_attn.v_kv_proj.parameters()} for k in hylayers]

    # load_model(model, "runs/2025-02-26/20:58:11", 0, hylayers)

    # 创建AdamW优化器
    optimizer = optim.AdamW(
        q2t_param + attn_param,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # 训练 quality brance
    trainq(model, val_datasets, optimizer, hylayers, gradient_accumulation_steps=1)

    # evaluate(model, val_datasets["test"]["konvid1k"])


if __name__ == "__main__":
    main()
    # TODO 词向量相似度
    # TODO Prompt-tuning
    # HYPERLAYER后添加LORA