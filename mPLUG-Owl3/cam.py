from threading import local
import cv2
from gradcam import GradCAM
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = "cuda:2" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
LLM = AutoModel.from_pretrained(
    "iic/mPLUG-Owl3-7B-241101",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    
    torch_dtype=torch.bfloat16,
).to(device)
tokenizer = AutoTokenizer.from_pretrained("iic/mPLUG-Owl3-7B-241101")


# Load the image
image_url = "/home/ippl/pxf/Diff-Plugin-modified/Diff-Plugin/CrossSet_IQA/data/kadid10k/images/I21_05_02.png"
image = Image.open(image_url).convert("RGB")
messages = [
    {"role": "user", "content": """<|image|>Is the picture beautiful?"""},
    {"role": "assistant", "content": ""}
]
processor = LLM.init_processor(tokenizer)
inputs = processor(messages, images=[image], videos=None).to(device)
with torch.no_grad():
    image_embeds = LLM.forward_image(inputs.pop("pixel_values"))
    inputs["image_embeds"] = image_embeds
tgt_logits = tokenizer.encode("Yes")
# 使用grad-cam画热力图
cam = GradCAM(arch=LLM.language_model, target_layer=LLM.language_model.model.layers[25].self_attn.v_kv_proj)
mask = cam(inputs, image, class_idx=9454)
# Convert the mask to a PIL image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode

def show_cam_on_image(img, mask, use_rgb=True):
    """
    Visualizes the Grad-CAM mask on the input image.

    Args:
        img (PIL.Image): The input image.
        mask (numpy.ndarray): The Grad-CAM mask.
        use_rgb (bool): Whether to convert the image to RGB format.

    Returns:
        PIL.Image: The image with the Grad-CAM mask applied.
    """
    # Resize the mask to match the input image size
    mask = np.uint8(mask * 255)
    mask = np.clip(mask, 0, 255)
    mask = np.transpose(mask, (1, 2, 0))
    mask = Image.fromarray(mask)
    mask = mask.resize(img.size, resample=InterpolationMode.BICUBIC)

    # Convert the image to numpy array
    img = np.array(img)

    # Apply the mask to the image
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    cam_image = heatmap + np.float32(img) / 255.0
    cam_image = cam_image / np.max(cam_image)

    return cam_image


mask = np.transpose(mask, (1, 2, 0))
cam_image = show_cam_on_image(image, mask)
plt.imshow(cam_image)
plt.axis("off")
plt.show()