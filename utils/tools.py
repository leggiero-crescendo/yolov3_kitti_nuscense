import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import tqdm
import torchvision

#parse model layer configuration
def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    module_defs = []
    type_name = None
    for line in lines:
        #print(line)
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name == "net":
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0            
        else:
            if type_name == "net":
                continue
            key,value = line.split('=')
            #value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


#parser the yolov3 configuration
def parse_hyperparm_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    module_defs = []
    for line in lines:
        #print(line)
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name != "net":
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name != "net":
                continue
            key,value = line.split('=')
            #value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def get_hyperparam(data):
    for d in data:
        if d['type'] == 'net':
            batch = int(d['batch'])
            subdivision = int(d['subdivisions'])
            momentum = float(d['momentum'])
            decay = float(d['decay'])
            saturation = float(d['saturation'])
            lr = float(d['learning_rate'])
            burn_in = int(d['burn_in'])
            max_batch = int(d['max_batches'])
            lr_policy = d['policy']
            in_width = int(d['width'])
            in_height = int(d['height'])
            in_channels = int(d['channels'])
            classes = int(d['class'])
            ignore_clas = int(d['ignore_cls'])

            return {'batch':batch,
                    'subdivision':subdivision,
                    'momentum':momentum,
                    'decay':decay,
                    'saturation':saturation,
                    'lr':lr,
                    'burn_in':burn_in,
                    'max_batch':max_batch,
                    'lr_policy':lr_policy,
                    'in_width':in_width,
                    'in_height':in_height,
                    'in_channels':in_channels,
                    'classes':classes,
                    'ignore_clas':ignore_clas }
        else:
            continue

def xywh2xyxy_np(x : np.array):
    y = np.zeros_like(x)
    y[...,0] = x[...,0] - x[...,2] / 2 #minx
    y[...,1] = x[...,1] - x[...,3] / 2 #miny
    y[...,2] = x[...,0] + x[...,2] / 2 #maxx
    y[...,3] = x[...,1] + x[...,3] / 2 #maxy
    return y

def drawBox(img):
    img = img * 255

    if img.shape[0] == 3:
        img_data = np.array(np.transpose(img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)

    #draw = ImageDraw.Draw(img_data)
    plt.imshow(img_data)
    plt.show()

#box_a, box_b IOU
#eps : 0으로 나누면 안되므로 0으로 나누는 것을 방지하기 위해 넣어주는것 
def bbox_iou(box_a, box_b, xyxy=False, eps = 1e-9):
    box_b = box_b.T

    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box_a[0], box_a[1], box_a[2], box_a[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box_b[0], box_b[1], box_b[2], box_b[3]
    else:
        b1_x1, b1_y1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
        b1_x2, b1_y2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
        b2_x1, b2_y1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
        b2_x2, b2_y2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2

    #intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    #union
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = b1_w * b1_h + b2_w * b2_h - inter + eps

    iou = inter / union

    return iou

def boxes_iou(box_a, box_b, xyxy=False, eps = 1e-9):
    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box_a[:,0], box_a[:,1], box_a[:,2], box_a[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box_b[:,0], box_b[:,1], box_b[:,2], box_b[:,3]
    else:
        b1_x1, b1_y1 = box_a[:,0] - box_a[:,2] / 2, box_a[:,1] - box_a[:,3] / 2
        b1_x2, b1_y2 = box_a[:,0] + box_a[:,2] / 2, box_a[:,1] + box_a[:,3] / 2
        b2_x1, b2_y1 = box_b[:,0] - box_b[:,2] / 2, box_b[:,1] - box_b[:,3] / 2
        b2_x2, b2_y2 = box_b[:,0] + box_b[:,2] / 2, box_b[:,1] + box_b[:,3] / 2

    #intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    #union
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = b1_w * b1_h + b2_w * b2_h - inter + eps

    iou = inter / union

    return iou

def cxcy2minmax(box):
    y = box.new(box.shape)
    xmin = box[...,0] - box[...,2] / 2
    ymin = box[...,1] - box[...,3] / 2
    xmax = box[...,0] + box[...,2] / 2
    ymax = box[...,1] + box[...,3] / 2
    y[...,0] = xmin
    y[...,1] = ymin
    y[...,2] = xmax
    y[...,3] = ymax

    return y

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.45):
    #num of class
    nc = prediction.shape[2] - 5
    #setting
    max_wh = 4096
    max_det = 300
    max_nms = 30000

    output = [torch.zeros((0,6), device='cpu')] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[x[..., 4] > conf_thresh]

        if not x.shape[0]:
            continue

        x[:,5:] *= x[:, 4:5] #class *= objectness

        box = cxcy2minmax(x[:,:4])

        conf, j = x[:,5:].max(1, keepdim=True)
        # print(conf, j)
        x = torch.cat((box, conf, j.float()), dim=1)[conf.view(-1) > conf_thresh]
        
        #number of boxes
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:,4].argsort(descending=True)[:max_nms]]

        c = x[:,5:6] * max_wh

        boxes, scores = x[:,:4] + c, x[:,4]

        i = torchvision.ops.nms(boxes, scores, iou_thresh)

        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()
    return output

def get_batch_statistics(predicts, targets, iou_threshold=0.5):
    batch_metrics = []
    for p in range(len(predicts)):
        if predicts[p] is None:
            continue
        predict = predicts[p]
        pred_boxes = predict[:,:4]
        pred_scores = predict[:,4]
        pred_labels = predict[:,-1]

        true_positives = np.zeros(pred_boxes.shape[0])
        
        annotations = targets[targets[:,0] == p][:,1:]
        target_labels = annotations[:,0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:,1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if len(detected_boxes) == len(annotations):
                    break
                if pred_label not in target_labels:
                    continue

                filtered_target_position, filtered_targets = zip(*filter(lambda x : target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # print(filtered_target_position, filtered_targets)
                # print(boxes_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)))

                iou, box_filtered_index = boxes_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                box_index = filtered_target_position[box_filtered_index]

                if iou > iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    print("unique_classes:", unique_classes)
    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap