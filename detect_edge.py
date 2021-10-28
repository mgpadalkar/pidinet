import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
from pathlib import Path
import numpy as np

import models
from models.convert_pidinet import convert_pidinet

def get_args():
  parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')
  parser.add_argument('--model', type=str, default='baseline', 
    help='model to train the dataset')
  parser.add_argument('--sa', action='store_true', 
    help='use CSAM in pidinet')
  parser.add_argument('--dil', action='store_true', 
    help='use CDCM in pidinet')
  parser.add_argument('--config', type=str, default='carv4', 
    help='model configurations, please refer to models/config.py for possible configurations')
  parser.add_argument('--checkpoint', type=str, default=None, 
    help='full path to checkpoint to be evaluated')
  parser.add_argument('--image', type=str, default='', 
    help='path of the image to be processed')
  parser.add_argument('--output_dir', type=str, default='./output', 
    help='output directory for saving the result')
  parser.add_argument('--resize_factor', type=float, default=1.0, 
    help='scale for resizing input')
  args = parser.parse_args()
  return args


def imagenet_tensor(img):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
  f = transforms.Compose([transforms.ToTensor(), normalize])
  return f(img)
 

def load_input(path, resize_factor):
  with open(path, 'rb') as f:
    img = Image.open(f)
    img = img.convert('RGB')
  img = imagenet_tensor(img)
  h = int(img.size(-2)*resize_factor)
  w = int(img.size(-1)*resize_factor)
  img = transforms.Resize((h, w))(img)
  return img.unsqueeze_(0)


def deprocess(result):
  out = torch.squeeze(result).cpu()
  out = out.numpy()
  out = Image.fromarray((out*255).astype(np.uint8))
  return out


def save_result(paths, model, device, resize_factor):
  img = load_input(paths[0], resize_factor)
  if img is None:
    print(f'Could not load image from {paths[0]}.')
    exit(-1)
  model.eval()
  with torch.no_grad():
    outputs = model(img.to(device))
  out     = deprocess(outputs[-1])
  name     = 'edge_' + Path(paths[0]).name
  filepath = Path(paths[1]).joinpath(name)
  Path(paths[1]).mkdir(parents=True, exist_ok=True)
  out.save(filepath)
  return [out, filepath]


def load_model(args):
  model = getattr(models, args.model)(args)
  model = torch.nn.DataParallel(model)
  loadinfo = "=> loading checkpoint from '{}'".format(args.checkpoint)
  state = torch.load(args.checkpoint, map_location='cpu')
  model.load_state_dict(convert_pidinet(state['state_dict'], args.config))
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  return [model, device]


if __name__ == '__main__':
  args = get_args()
  model, device = load_model(args)
  save_result([args.image, args.output_dir], model, device, args.resize_factor)
