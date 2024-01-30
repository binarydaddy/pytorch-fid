
from typing import Any
from PIL import Image
import torch
import torchvision.transforms as TF

class Collate_Dataset(object):
    def __init__(self, res=256) -> None:
        self.res = res
        self.transform = TF.Compose([
            TF.ToTensor()
        ])
        
    def __call__(self, batch) -> Any:
        batch = filter(lambda x: x is not None, batch)

        resized_images = []
        for idx, image in enumerate(batch):
            resized_image = image.resize((self.res, self.res), Image.BICUBIC)
            resized_images.append(self.transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        return image_tensors