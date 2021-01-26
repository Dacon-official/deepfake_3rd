import os
import glob
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from efficientnet import EfficientNet
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')

    # Ensemble model '{EffiecientNet_version}_{epoch}_{input_size}'
    parser.add_argument("--ens_model1", default='b7_22_550', help='First model to implement ensemble')
    parser.add_argument("--ens_model2", default='b6_35_500', help='Second model to implement ensemble')
    parser.add_argument("--ens_model3", default='b6_41_500', help='Third model to implement ensemble')
    parser.add_argument("--ens_model4", default='b5_47_400', help='Fourth model to implement ensemble')

    parser.add_argument("--data_root", default='./leaderboard')

    args = parser.parse_args()

    return args


def inference(args, device):

    ensemble_model = [args.ens_model1, args.ens_model2, args.ens_model3, args.ens_model4]

    for i, ens_model in enumerate(ensemble_model):

        efficientnet_version = ens_model.split('_')[0]
        epoch = ens_model.split('_')[1]
        input_size = int(ens_model.split('_')[2])

        if not os.path.exists("./csv_file"):
            os.makedirs('./csv_file', exist_ok=True)

        resume = f"./weights/{efficientnet_version}/{epoch}_{input_size}_checkpoint.pth.tar"
        save_csv = f"./csv_file/model{i+1}_inference.csv"

        sc = open(save_csv, 'w')
        sc.write('path,y\n')

        print(f"\n==> Inference model {i+1}")
        # load model
        print(f"==> Creating model EfficientNet {efficientnet_version}")
        model = EfficientNet.from_name(f'efficientnet-{efficientnet_version}')
        model = model.to(device)

        if os.path.isfile(resume):
            print(f"==> loading checkpoint '{resume}'")
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"==> loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"==> no checkpoint found at '{resume}'")

        cudnn.benchmark = True
        model = model.eval()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        images = glob.glob(os.path.join(args.data_root, '*.jpg'))
        transform = transforms.Compose([transforms.CenterCrop(input_size), transforms.ToTensor(), normalize])

        images.sort()
        m = nn.Softmax()

        with torch.no_grad():
            for j, image_path in tqdm(enumerate(images)):
                image = Image.open(image_path)
                image = transform(image)
                image = torch.unsqueeze(image, dim=0)
                image = image.to(device)

                output = model(image)
                output = m(output)[0]  # apply softmax

                image_tmpl = os.path.join('leaderboard', os.path.basename(image_path))

                # write to submission file
                if output[0] > output[1]:
                    saveline = image_tmpl + ',0'
                    sc.write(saveline)
                    sc.write('\n')
                else:
                    saveline = image_tmpl + ',1'
                    sc.write(saveline)
                    sc.write('\n')
        sc.close()


def ensemble(args):
    images = glob.glob(os.path.join(args.data_root, '*.jpg'))
    images.sort()

    y = np.zeros(len(images))

    for i in range(4):
        csv_file = open(f"./csv_file/model{i+1}_inference.csv", 'r')
        cvs_file_ = csv_file.readlines()
        csv_file.close()
        for j in range(len(images)):
            y[j] += int(cvs_file_[j+1].split(',')[1].split('\n')[0])

    y = np.round(y/4)

    final_csv = "./csv_file/test_submission.csv"
    sc = open(final_csv, 'w')
    sc.write('path,y')
    sc.write('\n')

    for i in range(len(images)):
        image_tmpl = os.path.join('leaderboard', os.path.basename(images[i]))
        saveline = image_tmpl + f',{int(y[i])}'
        sc.write(saveline)
        sc.write('\n')

    sc.close()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    args = parse_args()
    # inference(args, device)
    ensemble(args)
