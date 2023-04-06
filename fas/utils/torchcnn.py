import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

from fas.utils.model import load_checkpoint

class TorchCNN:
    '''Wrapper for torch model'''
    def __init__(self, model, checkpoint_path, config, device='cpu'):
        self.model = model
        if config.data_parallel.use_parallel:
            self.model = nn.DataParallel(self.model, **config.data_parallel.parallel_params)
        load_checkpoint(checkpoint_path, self.model, map_location=device, strict=True)
        self.config = config

    def preprocessing(self, images):
        ''' making image preprocessing for pytorch pipeline '''
        mean = np.array(object=self.config.img_norm_cfg.mean).reshape((3,1,1))
        std = np.array(object=self.config.img_norm_cfg.std).reshape((3,1,1))
        height, width = list(self.config.resize.values())
        preprocessed_imges = []
        for img in images:
            img = cv.resize(img, (height, width) , interpolation=cv.INTER_CUBIC)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = img/255
            img = (img - mean)/std
            preprocessed_imges.append(img)
        return torch.tensor(preprocessed_imges, dtype=torch.float32)

    def forward(self, batch):
        batch = self.preprocessing(batch)
        self.model.eval()
        model1 = (self.model.module
                  if self.config.data_parallel.use_parallel
                  else self.model)
        
        with torch.no_grad():
            output = model1.forward_to_onnx(batch)
            # features = self.model(batch)
            # output = model1.make_logits(features, all=False)
            return output.detach().numpy()
        