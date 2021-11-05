import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

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
  parser.add_argument('--checkpoint', type=str, default='', 
    help='full path to checkpoint to be evaluated')
  parser.add_argument('--image', type=str, default='', 
    help='path of the image to be processed')
  parser.add_argument('--output_dir', type=str, default='./output', 
    help='output directory for saving the result')
  parser.add_argument('--resize_factor', type=float, default=1.0, 
    help='scale for resizing input')
  args = parser.parse_args()
  return args


# https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


# def detect_edges(tile_image, tile_mask):
#   crack_label = np.stack(tile_image for _ in range(3)).transpose((1, 2, 0))
#   crack_label[tile_mask>0] = 255
#   return (10, crack_label)


def imagenet_tensor(img):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
  f = transforms.Compose([transforms.ToTensor(), normalize])
  return f(img)


def imread(path):
  with open(path, 'rb') as f:
    img = Image.open(f)
    img = img.convert('RGB')
    return img


def preprocess(img, resize_factor):
  if isinstance(img, np.ndarray):
    img = Image.fromarray(img).convert('RGB') 
  img = imagenet_tensor(img)
  h = int(img.size(-2)*resize_factor)
  w = int(img.size(-1)*resize_factor)
  img = transforms.Resize((h, w))(img)
  return img.unsqueeze_(0)


def deprocess(result):
  out = torch.squeeze(result).cpu()
  out = out.numpy()
  out = (out*255).astype(np.uint8)
  return out



def validate_checkpoint(ckpt):
  if os.path.exists(ckpt):
    print(f'==> Checkpoint found at \'{ckpt}\'')
    return ckpt
  # look for candidates in subdirectories
  cwd = os.getcwd()
  # print(f'Checkpoint \'{ckpt}\' not found.\n(Current working directory is \'{cwd}\')')
  candidates = [os.path.join(name, ckpt) for name in os.listdir(cwd)]
  found_at = [c for c in candidates if os.path.exists(c)]
  if len(found_at) > 0 and ckpt != '':
    print(f'==> Checkpoint found at \'{found_at[0]}\'')
    return found_at[0]
  # not found 
  print(f'==> Checkpoint NOT found at \'{ckpt}\' or in any subdirectories of \'{cwd}\'')
  return None


def load_model(args):
  if isinstance(args, dict):
    args = Bunch(args)
  ckpt = validate_checkpoint(args.checkpoint)
  if ckpt is None:
    return [None, None]
  model = getattr(models, args.model)(args)
  model = torch.nn.DataParallel(model)
  loadinfo = "==> loading checkpoint from '{}'".format(ckpt)
  print(loadinfo)
  state = torch.load(ckpt, map_location='cpu')
  model.load_state_dict(convert_pidinet(state['state_dict'], args.config))
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  return (model, device)

# def detect_edges(image, model, device, resize_factor):
#   gray = image[:, :, 0].copy()
#   gray = np.stack([gray, gray, gray], axis=2)
#   return gray

def detect_edges(image, model, device, resize_factor):
  if image is None:
    print(f'==> Input image is None; nothing to process.')
    return image
  img = preprocess(image, resize_factor)
  model.eval()
  with torch.no_grad():
    out = model(img.to(device))
  edge = deprocess(out[-1])
  # edge = np.stack([edge, edge, edge], axis=2) # check 3 channel
  return edge


def save_result(edge, image_name, save_path):
  if edge is None:
    return ''
  name = 'edge_' + image_name
  filepath = os.path.join(save_path, name)
  os.makedirs(save_path, exist_ok=True)
  out = Image.fromarray(edge)
  out.save(filepath)
  print(f'==> result saved at \'{filepath}\'')
  return filepath


def main(args):
  img = imread(args.image)
  model, device = load_model(args)
  edge = detect_edges(img, model, device, args.resize_factor)
  msg = save_result(edge, os.path.basename(args.image), args.output_dir)
  return msg

if __name__ == '__main__':
  args = get_args()
  main(args)
