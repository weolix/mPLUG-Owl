# data --LLM--> logits --linear-fitting--> label

import abc
from re import I
from threading import local
from IPython import embed
from PIL import Image
from numpy import indices
import torch.nn as nn
import torch, os
from typing import List
import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import csv
import matplotlib.pyplot as plt

data_path = {"kadid10k":"/home/ippl/zach/workspace/ntire/data/kadid10k/images",
             "spaq":"/home/ippl/zach/workspace/ntire/data/SPAQ/TestImage",
             "koniq10k":"/home/ippl/zach/workspace/ntire/data/koniq10k/1024x768",
             }
label_path = {"kadid10k":"/home/ippl/zach/workspace/ntire/data/kadid10k/dmos.csv",
              "spaq":"/home/ippl/zach/workspace/ntire/data/SPAQ/annotations/MOS and Image attribute scores.xlsx",
              "koniq10k":"/home/ippl/zach/workspace/ntire/data/koniq10k/koniq10k_scores_and_distributions.csv",
              }

model_path = "iic/mPLUG-Owl3-7B-241101"
device = "cuda:1"

# TODO 测试完善init函数加载数据
class spaq(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, label_path: str):
        self.img_dir = img_dir
        label_path = label_path
        self.image = []
        self.label = []
        data = pd.read_excel(label_path)
        # Convert DataFrame to dictionary with first column as keys
        self.data = data.to_dict(orient='index')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        path = os.path.join(self.img_dir, data["Image name"])
        return path, data["MOS"]


class kadid10k(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, label_path: str):
        self.img_dir = img_dir
        if label_path.endswith(".xlsx"):
            df = pd.read_excel(label_path)
        else:
            df = pd.read_csv(label_path)
        self.image = [os.path.join(img_dir, str(item)) for item in df.iloc[:, 0]]
        self.label = df.iloc[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.image[idx], self.label[idx]


class koniq10k(torch.utils.data.Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        
        if self.transform:
            image = self.transform(image)

        MOS = self.data.iloc[idx, 7]

        return img_name, MOS

class lsvq(torch.utils.data.Dataset):
    def __init__(self, label_path:str="/home/ippl/xxr/maxvqa/ExplainableVQA/examplar_data_labels/LSVQ/train_labels.txt"):
        self.label = pd.read_csv(label_path, header=None)[[0,3]]
        self.prefix = "/home/ippl/datasets/LSVQ/videos/"

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.prefix+self.label.iloc[idx, 0], self.label.iloc[idx, 1]
        

class Owl3logits(torch.nn.Module):
    def __init__(self, model_path: str):
        super(Owl3logits, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = self.model.init_processor(self.tokenizer)
    
    @torch.no_grad()
    def forward(self, img=None, video=None, token_with_very = False) -> torch.Tensor:
        media_token = "<|image|>" if img is not None else "<|video|>"
        if token_with_very:
            msg = [
                {
                    "role": "user",
                    "content": f"""{media_token}Analize from details, how would you rate the quality of this image?""",
                },
                {"role": "assistant", "content": "The quality of the image is very"},
            ]
        else:
            msg = [
                {
                    "role": "user",
                    "content": f"""{media_token}Analize from details, how would you rate the quality of this image?""",
                },
                {"role": "assistant", "content": "The quality of the image is"},
            ]
        inputs = self.processor(msg, images=[img], video=[video], preface=True).to(device)
        outputs = self.model(**inputs, output_hidden_states=True)
        logits = outputs.logits[:, -1]
        return logits

    def get_batch_logits(self, path_list) -> List[torch.Tensor]:

        logits = []
        for path in tqdm.tqdm(path_list, ncols=100):
            img = Image.open(path).convert('RGB')
            logits.append(self.forward(img).cpu())
        logits_all = torch.cat(logits)
        lmax = torch.max(logits_all, dim=0)
        top300max = torch.topk(lmax.values, 300, dim=-1)
        logits_top300 = logits_all[:,top300max.indices]
        logits_dict = {}
        for i, img in enumerate(path_list):
            logits_dict[img] = logits_top300[i]
        return logits_dict, top300max.indices
    
    def get_logits_with_indices(self, indices: list, path_list):
        logits_dict = {}
        for path in tqdm.tqdm(path_list, ncols=100):
            img = Image.open(path).convert('RGB')
            logits_dict[path] = self.forward(img)[0,indices].cpu()

        return logits_dict



def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    # rho = torch.mean(y_pred * y)
    # loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    # return ((loss0 + loss1) / 2).float()
    return loss0


def fit_linear(logits, linear, dataset, num_logits=150, lr=1e-3, workspace="./"):
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    os.chdir(workspace)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    linear = linear.to(device)

    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    max_rho = 0
    for epoch in range(100):
        linear.train()
        for img_path, label in train_loader:
            logitsi = torch.stack([logits[i] for i in img_path])
            logitsi = logitsi.to(device)
            label = label.float().to(device)
            pred = linear(logitsi[:, :num_logits])
            loss = plcc_loss(pred.squeeze(), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, lr: {scheduler.get_last_lr()[0]}, loss: {loss.item()}")

        linear.eval()
        with torch.no_grad():
            all_pred = []
            all_label = []
            for img_path, label in test_loader:
                logitsi = torch.stack([logits[i] for i in img_path])
                logitsi = logitsi.to(device)
                label = label.float().to(device)
                pred = linear(logitsi[:, :num_logits])
                all_pred.append(pred)
                all_label.append(label)
            all_pred = torch.cat(all_pred).squeeze(-1)
            all_label = torch.cat(all_label)
            srcc, _ = spearmanr(all_pred.cpu().numpy(), all_label.cpu().numpy())
            plcc, _ = pearsonr(all_pred.cpu().numpy(), all_label.cpu().numpy())
            rho = (srcc + plcc) / 2
            print(f"srcc: {srcc}, plcc: {plcc}")
            if rho > max_rho:
                max_rho = rho
                if epoch > 50:
                    if isinstance(linear, nn.Linear):
                        torch.save(linear.state_dict(), f"linear{num_logits}_{rho:.4f}.pth")
                    elif isinstance(linear, simple_attention):
                        torch.save(linear.state_dict(), f"attn{num_logits}_{rho:.4f}.pth")
        scheduler.step()


def list_path(img_dir: str, anno_file=None, format="png"):
    if anno_file:
        if anno_file.endswith('.csv'):
            df = pd.read_csv(anno_file)
        elif anno_file.endswith('.xlsx'):
            df = pd.read_excel(anno_file)
        path_list = [os.path.join(img_dir, str(item)) for item in df.iloc[:, 0]]
 
    else:
        img_list = os.listdir(img_dir)
        path_list = [
            os.path.join(img_dir, img)
            for img in img_list
            if img.endswith(format) and not img.startswith(".")
        ]

    return path_list

    


def test_cross(linear, logits, dataset, num_logits=200):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    linear.eval()
    with torch.no_grad():
        all_pred = []
        all_label = []
        for img_path, label in test_loader:
            logitsi = torch.stack([logits[i] for i in img_path])
            logitsi = logitsi.to(device)
            label = label.float().to(device)
            pred = linear(logitsi[:, :num_logits])
            all_pred.append(pred)
            all_label.append(label)
        all_pred = torch.cat(all_pred).squeeze()
        all_label = torch.cat(all_label)
        srcc, _ = spearmanr(all_pred.cpu().numpy(), all_label.cpu().numpy())
        plcc, _ = pearsonr(all_pred.cpu().numpy(), all_label.cpu().numpy())
        rho = (srcc + plcc) / 2
        print(f"rho: {rho}")

class simple_attention(nn.Module):
    def __init__(self, num_logits, num_heads=10, dropout=0.1, embed_dim=64):
        super(simple_attention, self).__init__()
        self.q_proj = nn.Linear(num_logits, embed_dim)
        self.k_proj = nn.Linear(num_logits, embed_dim)
        self.v_proj = nn.Linear(num_logits, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, logits):
        
        q = self.q_proj(logits)
        k = self.k_proj(logits)
        v = self.v_proj(logits)

        logits, _ = self.attention(q, k, v)
        scr = self.linear(logits)
        return scr
    

class MLP(nn.Module):
    def __init__(self, num_logits, hidden_size=256, output_size=1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(num_logits, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def main():
    os.chdir("exps")
    num_logits = 150
    spaq_set = spaq(data_path["spaq"], label_path["spaq"])
    kadid10k_set = kadid10k(data_path["kadid10k"], label_path["kadid10k"])
    # kadid10k_smallset = kadid10k(data_path["kadid10k"], "kadid10k/kadid2.2-2.3.xlsx", )
    koniq10k_set = koniq10k(data_path["koniq10k"], label_path["koniq10k"])
    # koniq10k_smallset = koniq10k(data_path["koniq10k"], "koniq10k/koniq3.4-3.7.csv")
    lsvq_set = lsvq()

    # # fit linear
    # linear = nn.Linear(num_logits, 1) 
    # logits = torch.load("lsvq/logits_lsvq.pt")
    # fit_linear(logits, linear, lsvq_set, num_logits, lr=1e-3, workspace="lsvq")

    # get logits
    os.chdir("../")
    owl3 = Owl3logits(model_path)
    logits_idc = owl3.get_batch_logits(list_path(data_path["koniq10k"], label_path["koniq10k"]))
    torch.save(logits_idc, "kadid10k/logits_discribe.pth")

    # # get logits with indices
    # indices = torch.load("kadid10k/ip_indices.pth")
    # owl3 = Owl3logits(model_path)
    # logits = owl3.get_logits_with_indices(indices, list_path(data_path["kadid10k"], "kadid10k/kadid2.2-2.3.xlsx"))
    # torch.save(logits, "logits_small_ip.pth")

    # # test cross spaq
    # linear = nn.Linear(num_logits, 1, device=device)
    # linear.load_state_dict(torch.load("spaq/linear150_0.9370.pth"))
    # logits, _ = torch.load("koniq10k/logits_dict.pth")
    # test_cross(linear, logits, koniq10k_set, num_logits)

    # # fit attention in kadid10k
    # logits = torch.load("logits.pth")
    # attn = simple_attention(num_logits, num_heads=1, dropout=0.1).to(device)
    # fit_linear(logits, attn, kadid10k_set, num_logits, lr=1e-4)

    # # fit MLP in kadid10k
    # logits = torch.load("logits.pth")
    # mlp = MLP(num_logits, hidden_size=64, output_size=1).to(device)
    # fit_linear(logits, mlp, kadid10k_set, num_logits, lr=2e-3)

    # # load indices
    # l, indices = torch.load("koniq10k/logits_discribe.pth")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # print(tokenizer.decode(indices))

    # kadid_idc = torch.load("kadid10k/ip_indices.pth")
    # koniq_idc = torch.load("koniq10k/indices.pth")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # count = 0
    # for i in range(300):
    #     if kadid_idc[i] in koniq_idc.tolist():
    #         print(tokenizer.decode(kadid_idc[i].item()))
    #         count += 1
    # print(count)
    
    # statedict=torch.load("kadid10k/linear300_0.9081.pth", map_location="cpu")["weight"].squeeze()
    # indices = torch.load("kadid10k/ip_indices.pth", map_location="cpu").tolist()
    # # print(statedict)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # weight = torch.tensor(list(zip(indices, statedict.tolist())))
    # indices_mask = abs(statedict)>0.01
    # wtd = weight[indices_mask.tolist()]

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # logits= torch.load("logits_small_ip.pth")
    # indice = torch.load("kadid10k/ip_indices.pth")
    # linear = torch.load("kadid10k/linear300_0.9093.pth", map_location="cpu")
    # score = {k: (logits[k]@linear["weight"].T).item() for k in logits.keys()}

    # # 结果保存到csv文件
    # with open("kadid10k/logits_first20.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     header = ["img", "distort", "level", "pred"] + [tokenizer.decode(indice[i]) for i in range(20)]
    #     writer.writerow(header)
    #     for k, v in logits.items():
    #         writer.writerow([k[-13:-10],k[-9:-7],k[-6:-4], f"{score[k]:.4}"] + v[:20].tolist())
    # # 聚类
    # n_clusters = 10
    # logit_keys = list(logits.keys())
    # logit_array = np.stack([logits[k].cpu().numpy() for k in logit_keys], axis=0)[:,:20]
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_clusters).fit(logit_array)
    # clusters = kmeans.labels_
    # for path, cluster_id in zip(logit_keys, clusters):
    #     print(path, cluster_id)
    # # 2D可视化
    # pca_data = PCA(n_components=2).fit_transform(logit_array)
    # plt.figure(figsize=(30,20), dpi=300)
    # plt.scatter(pca_data[:,0], pca_data[:,1], c=clusters, cmap='rainbow')
    # for i, label_text in enumerate(logit_keys):
    #     plt.annotate(label_text[-13:-4]+f" {score[label_text]:.4}", xy=(pca_data[i, 0], pca_data[i, 1]), fontsize=6, alpha=0.7)
    # plt.colorbar(label='Cluster')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('2D Visualization of Logits')
    # plt.savefig("pca.png")





if __name__ == "__main__":
    main()
    """
    lsvq indices:
    'poor good high low clear bad blurry very mon sharp, difficult accurate strange excellent unclear uns severe 
     fuzzy average serious inaccurate acceptable well extremely rich blurred unusual simple satisfactory dark precise
     poorly realistic detailed vivid important impressive ordinary chaotic差 limited decent problematic fine vibrant dim
     uneven close unreasonable weak rough dull poorest subjective beautiful mediocre vague obvious creative distorted biased
     crisp inferior favorable severely abnormal slightly hard black basic nice suitable un distinct odd over inadequate pleasing 
     strong slight reasonable little extreme abstract sub elegantgood terrible Poor artistic insufficient ind sparse incomplete 
     satisfying bright effective delicate soft interesting bl likely compromised significant successful dev informative excessive 
     badly plain evident generally complex clean p noticeable pass old refined small prominent.Poor lacking out minimal 
     disappointing unnatural natural grain monot peculiar unique fresh obscure common moderate sh uncertain even diverse large 
     appropriate unpleasant inappropriate desirable red general specific outstanding fragmented simplistic dynamic single heavy 
     def glo unfavorable seriously pixel intricate long positive unrealistic balanced promising easy professional similar clearly 
     t crude singular different straightforward questionable highly focused deficient dis pronounced visible mixed exaggerated 
     apparent heavily smooth yellow fragile commend sufficient overall complete substantial adequate好 challenging defined cool 
     comprehensive much ( diss varied great messy negative deep thin white coarse dire gradual exceptional distinctive normal 
     solid bizarre fair unhealthy ab ambiguous colorful detrimental ineffective shaky significantly dangerous thorough pleasant 
     authentic distantlow few gray useful attractive poorer Good discern concise confusing imaginative critical far sophisticated 
     satisfied expensive stable unacceptable sym unfair de superior relevant abundant clarity valuable l neat noticeably extensive 
     deviation好的 rare unreliable objective tight Low compelling outdated concerning crucial noisy skewed narrow preferred 
     striking lif uniform harsh light h blur unf remarkable best lack bland below intuitive green degraded pure recognizable 
     lower faint consistent cold'
    """
