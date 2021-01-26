import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models
import torchvision.datasets as dset
import random
import cv2
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchsummary import summary
import torch.utils.data as data
from tqdm import tqdm
import easydict
from dataset import DFDCDatatset
from efficientnet import EfficientNet
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='')
parser.add_argument('--train_list', type=str, default="new_train_list_1st.txt")
parser.add_argument('--test_list', type=str, default="new_test_list_1st.txt")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--test_batch_size', type=int, default=2)
parser.add_argument('--dist_weight', type=float, default=1)
parser.add_argument('--orig_weight', type=float, default=1)
parser.add_argument('--fgsm_weight', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--train_num_per_epoch', type=int, default=4000)
parser.add_argument('--test_num_per_epoch', type=int, default=2000)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')


logdir = os.path.join('./run','fgsm_b5')
summary_writer = SummaryWriter(logdir)

def fgsm_attack(model, loss, images, labels, eps) :
    
    images = images.cuda()
    labels = labels.cuda()
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels).cuda()
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

def main():
    global args
    args = parser.parse_args()
    print("Use GPU: {} for training".format(args.gpu))
    
    model = EfficientNet.from_pretrained('efficientnet-b5')
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    sym_criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, threshold=0.001, threshold_mode='rel', verbose=True)

    if args.resume:
        if os.path.isfile(args.resume):
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = DFDCDatatset(args.root,
                                 args.train_list,
                                 transforms.Compose([
                                     transforms.CenterCrop(400),
                                     transforms.ToTensor(),
                                     normalize,
                                 ]))

    test_dataset = DFDCDatatset(args.root,
                                args.test_list,
                                transforms.Compose([
                                    transforms.CenterCrop(400),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

    for epoch in range(args.start_epoch, args.epochs):

        part_tr = torch.utils.data.random_split(train_dataset, 
                        [args.train_num_per_epoch, len(train_dataset)-args.train_num_per_epoch])[0]
        part_te = torch.utils.data.random_split(test_dataset, 
                        [args.test_num_per_epoch, len(test_dataset)-args.test_num_per_epoch])[0]
        
        train_loader = DataLoader(part_tr, batch_size=args.train_batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)

        test_loader = DataLoader(part_te, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=False)
    
        train(train_loader, model, criterion, sym_criterion, optimizer, epoch, summary_writer)
        acc1 = validate(test_loader, model, criterion, epoch).cpu()
        acc2 = validate(test_loader, model, criterion, epoch, scheduler=scheduler, fgsm=True).cpu()
        acc = (acc1 + acc2)/2
        summary_writer.add_scalars('accuracy', {'val acc':acc1, 'fgsm acc':acc2}, epoch+1) 
        summary_writer.add_scalar('lr',optimizer.param_groups[0]['lr'], epoch+1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch+1)
        

def train(train_loader, model, criterion, sym_criterion, optimizer, epoch, summary_writer):
    model.train()
    eps=[0.007, 0.02, 0.05]

    print('-' * 50)
    print('Epoch {}/{}'.format(epoch + 1, args.epochs))

    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over data.
    correct = 0
    total = 0
    flag = 0
    
    for idx, (img, target) in tqdm(enumerate(train_loader)):

        fgsm_img = fgsm_attack(model, criterion, img, target, eps[idx%3]).cuda()
        img = img.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        out, feature = model(img, mode=True)
        fgsm_out, fgsm_feature = model(fgsm_img, mode=True)
        
        _, preds = torch.max(out.data, 1)
        _, fgsm_preds = torch.max(fgsm_out.data, 1)
        
        l2_loss = sym_criterion(feature, fgsm_feature)
        orig_loss = criterion(out, target)
        fgsm_loss = criterion(fgsm_out, target)

        correct += preds.eq(target.data.view_as(preds)).long().cpu().sum() + fgsm_preds.eq(target.data.view_as(fgsm_preds)).long().cpu().sum()
      
        loss = args.dist_weight*l2_loss + args.orig_weight*orig_loss + args.fgsm_weight*fgsm_loss
        total += img.size()[0]*2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            test_acc = 100. * correct / float(total)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} acc:{:.3f}%'.format(
                epoch + 1, idx * len(img), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item(), test_acc))
            print(f'l2_loss: {l2_loss},  orig_loss: {orig_loss}, fgsm_loss: {fgsm_loss}')
            correct = 0 
            total = 0
        
        summary_writer.add_scalar('train/loss', float(loss.item()), epoch+1)
        summary_writer.add_scalar('train/accuracy', float(test_acc), epoch+1)


def validate(test_loader, model, criterion, epoch, scheduler=None, fgsm=False):
    model.eval()
    eps=[0.007, 0.02]
    test_loss = 0
    correct = 0
    
    if fgsm is True:
        print("attack!")
        for idx, (images, target) in tqdm(enumerate(test_loader)):
            images = fgsm_attack(model, criterion, images, target, eps[idx%2]).cuda()
            target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            test_loss += criterion(output, target).item()
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            
        test_loss /= float(len(test_loader.dataset))
        test_acc = 100. * correct / float(len(test_loader.dataset))

        print('\nTest set: Average loss: {:.4f}, '
              'Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss,
                                                   correct, len(test_loader.dataset), test_acc))
        summary_writer.add_scalar('val/fgsm_loss', float(test_loss), epoch+1)
        scheduler.step(test_loss)
            
    else:
        with torch.no_grad():
            for idx, (images, target) in tqdm(enumerate(test_loader)):
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                output = model(images)
                test_loss += criterion(output, target).item()

                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

            test_loss /= float(len(test_loader.dataset))
            test_acc = 100. * correct / float(len(test_loader.dataset))

            print('\nTest set: Average loss: {:.4f}, '
                  'Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss,
                                                       correct, len(test_loader.dataset), test_acc))
            summary_writer.add_scalar('val/basic_loss', float(test_loss), epoch+1)
        
    return test_acc


def save_checkpoint(state, epoch):
    path = os.path.join('./weights', 'b5')
    os.makedirs(path, exist_ok=True)
    filename=os.path.join(path,'{}_400_checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)



if __name__ == '__main__':
    main()
