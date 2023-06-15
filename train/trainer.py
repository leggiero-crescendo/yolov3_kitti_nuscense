import os, sys
import torch
import torch.optim as optim

from utils.tools import *
from train.loss import *

from terminaltables import AsciiTable

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam, device, torch_writer):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batch']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yololoss = Yololoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'])

        self.scheduler_mulistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[20,40,60],
                                                            gamma = 0.5)
        self.torch_writer = torch_writer

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            #drop the batch when invalid values
            if batch is None:
                continue
            input_img, targets, anno_path = batch
            # print("input : ", input_img.shape, targets.shape)
            input_img = input_img.to(self.device, non_blocking=True)
            output = self.model(input_img)
            # print("output - len : {}, shape : {}".format(len(output),output[0].shape))
            # get loss between output and target
            loss, loss_list = self.yololoss.compute_loss(output, targets, self.model.yolo_layers)

            # get gradients
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler_mulistep.step(self.iter)
            self.iter += 1
            # loss_list = [loss.item(), lobj.item(), lcls.item(), lbox.item()]
            loss_name = ['total_loss', 'obj_loss', 'cls_loss', 'box_loss']

            if i % 10 == 0:
                print("epoch {} / iter {} lr-learning rate {} loss {}".format(self.epoch, self.iter,get_lr(self.optimizer), loss.item()))
                self.torch_writer.add_scalar('lr', get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('total_loss', loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)

        return loss
            

    def run_eval(self):
        predict_all = []
        gt_labels = []
        for i, batch in enumerate(self.eval_loader):
            if batch is None:
                continue
            input_img, targets, anno_path = batch
            input_img = input_img.to(self.device, non_blocking=True)
            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(prediction=output, conf_thresh=0.4, iou_thresh=0.45)
                print("eval output : ", output.shape, " best_box_list : ", len(best_box_list), best_box_list[0].shape)

            gt_labels += targets[...,1].tolist()
            targets[..., 2:6] = cxcy2minmax(targets[...,2:6])
            input_wh = torch.tensor([input_img.shape[3], input_img.shape[2], input_img.shape[3], input_img.shape[2]]) #whwh
            targets[...,2:6] *= input_wh
            # print(targets[...,2:6])

            predict_all += get_batch_statistics(best_box_list, targets, iou_threshold=0.5)

            # print(predict_all)
        true_positives, pred_scores, pred_labels = [np.concatenate(x,0) for x in list(zip(*predict_all))]

        #get map, recalls
        matrics_output = ap_per_class(true_positives, pred_scores, pred_labels, gt_labels)
        # print(matrics_output)
        if matrics_output is not None:
            precision, recall, ap, f1, ap_class = matrics_output
            ap_table = [['install', 'ap']]
            for i, c in enumerate(ap_class):
                ap_table += [[c,"%.5f" % ap[i]]]

            # print(AsciiTable(ap_table).table)



            # input_whwh = torch.tensor([input_img.shape[3], input_img.shape[2], input_img.shape[3], input_img.shape[2]])
            # for idx in range(input_img.shape[0]):
            #     print(targets[targets[:,0] == idx][2:6].shape, input_whwh.shape)
            #     drawBox(input_img[idx], box = best_box_list[0], xyxy=True)
               
        return


    def run(self):
        while True:
            if self.max_batch <= self.iter:
                break            

            self.model.train()
            #loss calculation
            loss = self.run_iter()
            self.epoch += 1

            #evaluation
            self.model.eval()
            self.run_eval()

            #save model (checkpoint)
            checkpoint_path = os.path.join("./output", "model_epoch"+str(self.epoch)+".pth")
            torch.save({'epoch' : self.epoch,
                        'iteration' : self.iter,
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer_state_dict' : self.optimizer.state_dict(),
                         'loss' : loss }, checkpoint_path)
            
            