import os, sys
import torch
import argparse
from torch.utils.data.dataloader import DataLoader

from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transforms import *
from model.yolov3 import *
from train.trainer import *

from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH argumants")
    parser.add_argument("--gpus", type=int, nargs='+', default=[], help="List of GPU device id")
    parser.add_argument("--mode", type=str, help="mode : train / eval / demo", default=None)
    parser.add_argument("--cfg", type=str, help="model config path", default=None)
    parser.add_argument('--checkpoint', type=str, help="model checkpoint", default=None, )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    #skip invalid data
    if len(batch) == 0:
        return
    imgs, targets, anno_path = list(zip(*batch))
    imgs = torch.stack([img for img in imgs]) #dim3=>dim4
    for i, boxes in enumerate(targets):
        #insert index of batch
        boxes[:,0] = i
        # print(boxes)

    targets = torch.cat(targets,0)

    return imgs, targets, anno_path


def train(cfg_param=None, using_gpus=None):
    print("training")
    #dataloader 6081 images / batch 4
    my_transform = get_transfromations(cfg_param=cfg_param, is_train=True)
    train_data = Yolodata(is_train=True, 
                          transform=my_transform, 
                          cfg_param=cfg_param)
    train_loader = DataLoader(train_data,
                              batch_size = cfg_param['batch'],
                              num_workers = 0,
                              pin_memory = True,
                              drop_last = True,
                              shuffle = True,
                              collate_fn=collate_fn)

    eval_transform = get_transfromations(cfg_param=cfg_param, is_train=False)
    eval_data = Yolodata(is_train=False,
                         transform=eval_transform,
                         cfg_param=cfg_param)
    eval_loader = DataLoader(eval_data,
                             batch_size = cfg_param['batch'],
                              num_workers = 0,
                              pin_memory = True,
                              drop_last = False,
                              shuffle = False,
                              collate_fn=collate_fn)
            
    model = Darknet53(args.cfg, cfg_param)
    model.train()
    model.initialize_weights()
    
    #set device
    #print("GPU : ",torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    #load checkpoint
    #If checkpoint is existed, load the previous checkpoint.
    if args.checkpoint is not None:
        # print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # print(checkpoint)
        # for key, value in checkpoint['model_state_dict'].items():
        #     print(key, value)
        model.load_state_dict(checkpoint['model_state_dict'])


    # sys.exit(1)

    torch_writer = SummaryWriter("./output")

    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, hparam=cfg_param, device=device, torch_writer = torch_writer)
    trainer.run()

    # for i, batch in enumerate(train_loader):
    #     img, targets, anno_path = batch

    #     output = model(img)

        # print("output len : {}, 0th shape;{}".format(len(output), output[0].shape))
        # sys.exit(1)

    #for name, param in model.named_parameters():
    #    print(f"name : {name}, shape {param.shape}")
    
    # for i, batch in enumerate(train_loader):
    #     #print(i, len(batch))
    #     #print(batch)
    #     img, targets, anno_path = batch
    #     #print("iter {}, img {}, targets {}, anno_path {}".format(i, img.shape, targets.shape, anno_path))
    #     drawBox(img[0].detach().cpu())

def eval(cfg_param=None, using_gpus=None):
    print("evauation")

def demo(cfg_param=None, using_gpus=None):
    print("demo")

if __name__ == "__main__":
    print("main")
    args = parse_args()
    #print("args : ", args)
    #print(args.mode, args.gpus)

    #cfg parser
    net_data = parse_hyperparm_config(args.cfg)
    #print(net_data)
    cfg_param = get_hyperparam(net_data)
    # print(cfg_param)

    using_gpus = [int(g) for g in args.gpus]

    if args.mode == "train":
        #training
        train(cfg_param = cfg_param, using_gpus = using_gpus)
    elif args.mode == "eval":
        #evaluation
        eval(cfg_param=cfg_param)
    elif args.mode == "demo":
        #demo
        demo(cfg_param=cfg_param)
    else:
        print("unknown mode")
    
    print("finish")
