# 输出csv中的图像的logits
from PIL import Image
from owl3_zeroshot import MultimodalQualityEvaluator

def log_diff_images(csv_file):
    """
    输出两张图像的logits差异
    :param image1: 第一张图像
    :param image2: 第二张图像
    """
    evaluator = MultimodalQualityEvaluator()
    for image_name in csv_file['image_name']:
        image = Image.open(image_name).convert('RGB')
        image_name = image_name.split('/')[-1]
        output = evaluator.forward(image)
        
        logits[image_name] = output['logits']
    
    return logits