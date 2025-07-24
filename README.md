<div style="display: flex; align-items: center;">
  <h1>mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models</h1>
</div>

<div align="center">
Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, Jingren Zhou
</div>
<div align="center">
<strong>Tongyi, Alibaba Group</strong>
</div>

<div align="center">
    <a href="https://github.com/X-PLUG/mPLUG-Owl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://arxiv.org/abs/2408.04840"><img src="https://img.shields.io/badge/Paper-ArXiv-orange.svg" ></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FmPLUG-Owl&count_bg=%23E97EBA&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false" alt="Hits"></a>
</div>

---

**mPLUG-Owl3** is the latest version of the mPLUG-Owl series, featuring advanced long image-sequence understanding capabilities in Multi-Modal Large Language Models.

![Performance and Efficiency](./mPLUG-Owl3/assets/performance.png)

## News and Updates
* ```2024.11.27``` 🔥🔥🔥 We have released the latest version of [mPLUG-Owl3-7B-241101](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-241101). The performance in video and multi-image scenarios is significantly improved. It also achieves top-1 performance on [LVBench](https://lvbench.github.io/)🎉🎉🎉.
* ```2024.10.15``` We have released small-sized models of mPLUG-Owl3 based on the 0.5B and 1.5B Qwen2. [Checkpoints](#Checkpoints) are available on ModelScope and HuggingFace. Now you can experience Owl3's ultra-long visual content comprehension on edge devices.
* ```2024.09.23``` Thanks to ms-swift. The finetuning of mPLUG-Owl3 is now supported. Refer to the document at [Finetuning of mPLUG-Owl3](https://github.com/modelscope/ms-swift/issues/1969#issue-2511475059).
* ```2024.09.23``` We have released the evaluation pipeline, which can be found at [Evaluation](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl3/evaluation). Please refer to the README for more details.
* ```2024.08.12``` We release **mPLUG-Owl3**. The source code and weights are avaliable at [HuggingFace](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728).

## Cases
mPLUG-Owl3 can learn from knowledge from retrieval system.

![RAG ability](./mPLUG-Owl3/assets/fewshot.png)

mPLUG-Owl3 can also chat with user with a interleaved image-text context.

![Interleaved image-text Dialogue](./mPLUG-Owl3/assets/multiturn.png)

mPLUG-Owl3 can watch long videos such as movies and remember its details. 

![Long video understanding](./mPLUG-Owl3/assets/movie.png)

## TODO List
- [x] Evaluation with huggingface model.
- [ ] Training data releasing. All training data are sourced from the public datasets. We are preparing to release a compact version to facilitate easy training. Prior to this release, you have the option to manually organize the training data.
- [x] Training pipeline.

## Performance

Visual Question Answering
![VQA](./mPLUG-Owl3/assets/vqa.png)

Multimodal LLM Benchmarks
![Multimodal Benchmarks](./mPLUG-Owl3/assets/mmb.png)

Video Benchmarks
![Video Benchmarks](./mPLUG-Owl3/assets/video_bench.png)

Multi-image Benchmarks
![Multiimage Benchmarks](./mPLUG-Owl3/assets/multiimage_bench.png)
![MI-Bench](./mPLUG-Owl3/assets/mibench.png)

### The comparison between mPLUG-Owl3-7B-240728 and mPLUG-Owl3-7B-241101

| Model                |NextQA |MVBench |VideoMME w/o sub| LongVideoBench-val| MLVU| LVBench|
|-|-|-|-|-|-|-|
| mPLUG-Owl3-7B-240728| 78.6 |54.5 |53.5 |52.1 |63.7|-|
| mPLUG-Owl3-7B-241101|82.3|59.5|59.3	|59.7|70.0|43.5|

| Model                |NLVR2 |Mantis-Eval |MathVerse-mv| SciVerse-mv| BLINK |Q-Bench2|
|-|-|-|-|-|-|-|
| mPLUG-Owl3-7B-240728| 90.8 |63.1 |65.0 |86.2 |50.3 |74.0|
| mPLUG-Owl3-7B-241101|92.7|67.3|65.1	|82.7|53.8|77.7|

| Model |VQAv2	 | OK-VQA	| GQA	| VizWizQA	| TextVQA	|
|-|-|-|-|-|-|
| mPLUG-Owl3-7B-240728|82.1 |60.1| 65.0| 63.5 |69.0|
| mPLUG-Owl3-7B-241101|83.2 |61.4| 64.7| 62.9 |71.4|

| Model | MMB-EN 	|MMB-CN 	|MM-Vet 	|POPE 	|AI2D|
|-|-|-|-|-|-|
| mPLUG-Owl3-7B-240728|77.6 |74.3   |40.1 |88.2 |73.8|
| mPLUG-Owl3-7B-241101|80.4	|79.1	|39.8	|88.1	|77.8|

## Evaluation

To perform evaluation on the above benchmarks, first download the datasets from the official or huggingface sites: [ai2d](https://huggingface.co/datasets/lmms-lab/ai2d), [gqa](https://cs.stanford.edu/people/dorarad/gqa/download.html), [LLaVA-NeXT-Interleave-Bench](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Interleave-Bench), [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench), [mmbench](https://github.com/open-compass/MMBench), [mmvet](https://github.com/yuweihao/MM-Vet), [mvbench](https://huggingface.co/datasets/OpenGVLab/MVBench), [nextqa](https://huggingface.co/datasets/lmms-lab/NExTQA), [NLVR2](https://huggingface.co/datasets/lmms-lab/NLVR2), [okvqa](https://okvqa.allenai.org/download.html), [qbench2](https://huggingface.co/datasets/q-future/Q-Bench2-HF), [textvqa](https://textvqa.org/), [videomme](https://huggingface.co/datasets/lmms-lab/Video-MME), [vizwiz](https://vizwiz.org/tasks-and-datasets/vqa/), [vqav2](https://visualqa.org/download.html).

Then organize them as follows in `./mPLUG-Owl3/evaluation/dataset`.

We provide the json files of some datasets [here](https://huggingface.co/datasets/StarBottle/mPLUG-Owl3-Evaluation), to help reproduce the evaluation results in our paper.

Once the data is ready, run `./mPLUG-Owl3/evaluation/eval.sh` for evaluation.
The datasets configuration can be modified in `./mPLUG-Owl3/evaluation/tasks/plans/all.yaml`.

## Checkpoints
<span id="Checkpoints"></span>
| Model Size | ModelScope | HuggingFace |
|------------|------------|-------------|
| 1B         | [mPLUG-Owl3-1B-241014](https://modelscope.cn/models/iic/mPLUG-Owl3-1B-241014) | [mPLUG-Owl3-1B-241014](https://huggingface.co/mPLUG/mPLUG-Owl3-1B-241014) |
| 2B         | [mPLUG-Owl3-2B-241014](https://modelscope.cn/models/iic/mPLUG-Owl3-2B-241014) | [mPLUG-Owl3-2B-241014](https://huggingface.co/mPLUG/mPLUG-Owl3-2B-241014) |
| 7B         | [mPLUG-Owl3-7B-240728](https://modelscope.cn/models/iic/mPLUG-Owl3-7B-240728) | [mPLUG-Owl3-7B-240728](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728) |
| 7B         | - | [mPLUG-Owl3-7B-241101](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-241101) |

## Usage

### Gradio Demo
Installing the dependencies
```
cd mPLUG-Owl3
pip install -r requirements.txt
```

Execute the demo.
```
python mPLUG-Owl3/gradio_demo.py
```

### Quickstart
#### The models after 241101
Load the mPLUG-Owl3. We now only support attn_implementation in ```['sdpa', 'flash_attention_2']```.
```Python
import torch
from modelscope import AutoConfig, AutoModel
model_path = 'iic/mPLUG-Owl3-2B-241101'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(config)
model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True)
_ = model.eval().cuda()
device = "cuda"
```

Chat with images.
```Python
from PIL import Image

from modelscope import AutoTokenizer
from decord import VideoReader, cpu 
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

image = Image.new('RGB', (500, 500), color='red')

messages = [
    {"role": "user", "content": """<|image|>
Describe this image."""},
    {"role": "assistant", "content": ""}
]

inputs = processor(messages, images=[image], videos=None)

inputs.to('cuda')
inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens':100,
    'decode_text':True,
})

g = model.generate(**inputs)
print(g)
```

Chat with a video.
```Python
from PIL import Image

from modelscope import AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

messages = [
    {"role": "user", "content": """<|video|>
Describe this video."""},
    {"role": "assistant", "content": ""}
]

videos = ['/path/to/your/video.mp4']

MAX_NUM_FRAMES=16

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames
video_frames = [encode_video(_) for _ in videos]
inputs = processor(messages, images=None, videos=video_frames)

inputs.to(device)
inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens':100,
    'decode_text':True,
})

g = model.generate(**inputs)
print(g)
```

#### Save memory by Liger-Kernel
mPLUG-Owl3 is based on Qwen2, which can be optimized through the Liger-Kernel to reduce memory usage.
```
pip install liger-kernel
```

#### Save memory by setting device_map
When you have more than one GPUs, you can set the ```device_map='auto'``` to split the mPLUG-Owl3 into multiple GPUs. However, it will slowdown the inference speed.

```python
model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
_ = model.eval()
first_layer_name = list(model.hf_device_map.keys())[0]
device = model.hf_device_map[first_layer_name]
```

### The models before 241101
Load the mPLUG-Owl3. We now only support attn_implementation in ```['sdpa', 'flash_attention_2']```.
```Python
import torch
from transformers import AutoConfig, AutoModel
model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
model.eval().cuda()
```

## Finetuning
Please use ms-swift to finetuning the mPLUG-Owl3. Here is an [instruction](https://github.com/modelscope/ms-swift/issues/1969).

**For mPLUG-Owl3-7B-241101 and newer versions, you should set the ```model_type``` to ```mplug-owl3v-7b-chat``` instead.**

## Citation

If you find mPLUG-Owl3 useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{ye2024mplugowl3longimagesequenceunderstanding,
      title={mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models},
      author={Jiabo Ye and Haiyang Xu and Haowei Liu and Anwen Hu and Ming Yan and Qi Qian and Ji Zhang and Fei Huang and Jingren Zhou},
      year={2024},
      eprint={2408.04840},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04840},
}
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for the authors of LLaVA for providing the framework.

## Related Projects

* [LLaMA](https://github.com/facebookresearch/llama). A open-source collection of state-of-the-art large pre-trained language models.
* [LLaVA](https://github.com/haotian-liu/LLaVA). A visual instruction tuned vision language model which achieves GPT4 level capabilities.
* [mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG). A vision-language foundation model for both cross-modal understanding and generation.
* [mPLUG-2](https://github.com/alibaba/AliceMind). A multimodal model with a modular design, which inspired our project.
