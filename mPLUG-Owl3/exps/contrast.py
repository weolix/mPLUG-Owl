from PIL import Image
import torch.nn as nn
import torch
from typing import List
from transformers import AutoTokenizer, AutoProcessor, AutoModel
import transformers


class OwlDiscriminator(nn.Module):
    def __init__(self, model_path="iic/mPLUG-Owl3-7B-241101", device="cuda"):
        super().__init__()

        model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device=device)
        model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = model.init_processor(self.tokenizer)

        self.msg = [
            {
                "role": "user",
                # "content": """<|image|><|image|>Analize from details, is the quality of the first image better than the second? Say "Yes" or "No". If they can't be identified, say "The same".""",
                "content": """<|image|><|image|>Analize from details, which image has better quality? Say "First" or "Second". If they can't be identified, say "Same".""",
            },
            {"role": "assistant", "content": ""},
        ]

        # self.preferential_ids = [id_ for id_ in self.tokenizer(["Yes", "The", "No"])["input_ids"]]
        self.preferential_ids = [id_ for id_ in self.tokenizer(["First", "Same", "Second"])["input_ids"]]
        self.weight = torch.tensor([1.0, 0.0, -1.0])


        self.model = model
        self.dev = device


    def forward(self, image: List[Image.Image]):
        with torch.inference_mode():
            input = self.processor(self.msg, images=image, videos=None).to(self.dev)
            output_logits = self.model(**input).logits

            predict_logits = output_logits[:,-1,self.preferential_ids]
            print(self.tokenizer.decode(output_logits.argmax(-1)[0,-1]))
            odds = torch.softmax(predict_logits[0,:,0], -1).cpu()
            return odds, odds @ self.weight


class Owlscore(nn.Module):
    def __init__(self, model_path="iic/mPLUG-Owl3-7B-241101", device="cuda"):
        super().__init__()

        model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device=device)
        model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = model.init_processor(self.tokenizer)

        self.msg = [
            {
                "role": "user",
                "content": """<|image|>Analize from details, how would you rate the quality of this image?""",
            },
            {"role": "assistant", "content": "The quality of the image is very"},
        ]

        self.model = model
        self.dev = device


    def forward(self, image: List[Image.Image]):
        with torch.inference_mode():
            input = self.processor(self.msg, images=image, videos=None, preface=True).to(self.dev)
            output_logits = self.model(**input).logits
            topk = output_logits[0,-1].topk(k=200)
            odds = torch.softmax(topk.values, -1).cpu()
            print("shape: ",output_logits.shape)
            # print(self.tokenizer.decode(topk.indices))

            return odds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="iic/mPLUG-Owl3-7B-241101")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_path0", type=str, default="/home/ippl/xxr/DDPG/exp/image_samples/imagenet_samples/4_0.png")
    parser.add_argument("--img_path1", type=str, default="/home/ippl/xxr/DDPG/exp/image_samples/imagenet_samples/Apy/orig_4.png")
    args = parser.parse_args()

    scorer = Owlscore(model_path=args.model_path, device=args.device)
    score = scorer(
        [
            Image.open(args.img_path1),
        ]
    )
    print(score)
