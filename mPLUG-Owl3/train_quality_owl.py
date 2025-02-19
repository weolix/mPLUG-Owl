import os
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1" # 可选，去除颜色编码
os.environ["AV_LOG_QUIET"] = "1" # 设置日志级别为 quiet
import warnings
def ignore_h264_warning(message, category, filename, lineno, file=None, line=None):
    return True if "mmco: unref short failure" in str(message) else False
warnings.filterwarnings("ignore", message=".*mmco: unref short failure.*")


import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from dataset import ViewDecompositionDataset, dict_simply_collate
from quality_plugowl3 import QualityOwl3Model
import yaml
import tqdm, time
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter


def trainq(model, dataloader, valdataloader, optimizer, opt):
    epochs = opt.get("epochs", 10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    date_str = time.strftime("%Y-%m-%d", time.localtime())
    time_str = time.strftime("%H:%M:%S", time.localtime())
    writer = SummaryWriter(opt.get("logging_dir", f"runs/{date_str}/{time_str}"))

    global_step = 0
    best_plcc = 0.7
    for epoch in range(epochs):
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        # 验证
        model.eval()
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
            writer.add_scalar("Spearmanr/val", spearmanrcc[0], epoch)
            writer.add_scalar("Pearsonr/val", pearsonrcc[0], epoch)
            print(f"eval Spearmanr: {spearmanrcc[0]:.4f}, Pearsonr: {pearsonrcc[0]:.4f}")
            if spearmanrcc[0] > best_plcc:
                best_plcc = spearmanrcc[0]
                torch.save(model.LLM.language_model.model.layers[26].self_attn.v_kv_proj, f"attn{epoch}.pth")
                torch.save(model.quality2text_model, f"q2t{epoch}.pth")
                print(f"Best model saved at epoch {epoch+1}")

        # 训练
        model.train()
        total_loss = 0
        pred_scores = []
        gt_scores = []
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)):
            aesthetic = batch["aesthetic"]
            technical = batch["technical"] 
            labels = batch["gt_label"].to(model.LLM.device)

            outputs = model(aesthetic=aesthetic, technical=technical, labels=labels)
            plccloss, rankloss, score = outputs.loss
            loss = plccloss #+ rankloss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            optimizer.zero_grad()

            # 记录loss后立即释放
            current_loss = loss.item()
            writer.add_scalar("Loss/step", current_loss, global_step)
            global_step += 1

            pred_scores.extend(score.detach().cpu().tolist())
            gt_scores.extend(labels.detach().cpu().tolist())

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


        scheduler.step()
    writer.close()



def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载配置文件，假设 data.yml 中包含训练参数
    opt = yaml.safe_load(open("data.yml", "r"))
    tdataset = ViewDecompositionDataset(opt["train"])
    vdataset = ViewDecompositionDataset(opt["val"])
    tdataloader = DataLoader(tdataset, 
                            batch_size=5, 
                            num_workers=8, 
                            shuffle=True, 
                            collate_fn=dict_simply_collate,
                            pin_memory=True
                            )
    vdataloader = DataLoader(vdataset,
                            batch_size=5,
                            num_workers=8,
                            shuffle=False,
                            collate_fn=dict_simply_collate,
                            pin_memory=True
                            )
    


    # 根据需要设置设备

    # 创建模型，启用 quality brance 进行训练
    model = QualityOwl3Model(tech_brance=True).to(device)
    parameters_to_optimize = [
    {'params': model.LLM.language_model.model.layers[26].self_attn.v_kv_proj.parameters()},
    {'params': model.quality2text_model.parameters()}
    ]

    model.LLM.language_model.model.layers[26].self_attn.v_kv_proj.load_state_dict(torch.load("attn5.pth").state_dict())
    model.quality2text_model.load_state_dict(torch.load("q2t5.pth").state_dict())
    # 创建AdamW优化器
    optimizer = optim.AdamW(
        parameters_to_optimize,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # 训练 quality brance
    trainq(model, tdataloader, vdataloader, optimizer, opt)


if __name__ == "__main__":
    main()