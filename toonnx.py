import os
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model import MobileNet2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model    
    model = MobileNet2(scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6).to(device)
    weights_path = 'results/RSCD/model_best.pth'
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
    # state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    # load model weights
    # weights_path = r"./MobileNetV2-master/results/mobilenetv2_RSCD.pth"
    # weights_path = r"./result/model_best.pth"
    # assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    # model.load_state_dict(torch.load(weights_path, map_location=device))
   
    model.eval()
    input_names = ['input']
    output_names = ['output']
    x = torch.randn(1, 3, 224, 224).to(device)
    #x = torch.randn(1, 3, 375, 1242).cuda() #kitti
    torch.onnx.export(model, x, 'mobilenetv2_RSCD.onnx',opset_version=11,  input_names=input_names, output_names=output_names,
                      verbose='False')




if __name__ == '__main__':
    main()

