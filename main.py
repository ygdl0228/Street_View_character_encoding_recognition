# @Time    : 2023/4/22 20:43
# @Author  : ygd
# @FileName: main.py
# @Software: PyCharm

from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import glob
import json
import multiprocessing as mp
import torchvision.transforms as transforms
import time
from torch import nn
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)


class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5


def train(train_loader, model, criterion, optimizer, device):
    model.to(device)
    model.train()
    train_loss = []
    for i, (input, target) in enumerate(train_loader):
        input.to(device)
        target.to(device)
        c0, c1, c2, c3, c4, = model(input)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3]) + \
               criterion(c4, target[:, 4])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


def val(val_loader, model, criterion, device):
    model.to(device)
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            c0, c1, c2, c3, c4 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3]) + \
                   criterion(c4, target[:, 4])
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, device, tta=10):
    model.to(device)
    model.eval()
    test_pred_tta = None
    # TTA 次数
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                input = input.to(device)
                c0, c1, c2, c3, c4 = model(input)
                output = np.concatenate([
                    c0.data.cpu().numpy(),
                    c1.data.cpu().numpy(),
                    c2.data.cpu().numpy(),
                    c3.data.cpu().numpy(),
                    c4.data.cpu().numpy()], axis=1)
            test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

        return test_pred_tta


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
        "cpu")
    train_path = glob.glob('./train/*.png')
    train_path.sort()
    train_json = json.load(open('./train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path,
                    train_label,
                    transforms.Compose([transforms.Resize((64, 128)),
                                        transforms.RandomCrop((60, 120)),
                                        transforms.ColorJitter(0.3, 0.3, 0.2),
                                        transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
        batch_size=40,
        shuffle=True,
        num_workers=14
    )
    val_path = glob.glob('../val/*png')
    val_path.sort()
    val_json = json.load(open('val.json'))
    val_label = [val_json[x]['label'] for x in val_json]
    val_loader = torch.utils.data.DataLoader(
        SVHNDataset(val_path,
                    val_label,
                    transforms.Compose([transforms.Resize((60, 120)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])),
        batch_size=40,
        shuffle=False,
        num_workers=14
    )
    test_path = glob.glob('./test/*.png')
    test_path.sort()
    test_label = [[1]] * len(test_path)
    test_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_path,
                    test_label,
                    transforms.Compose([transforms.Resize((70, 140)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])),
        batch_size=40,
        shuffle=False,
        num_workers=14,
    )
    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    best_loss = 1000.0
    # 是否使用GPU

    for epoch in range(10):
        train_loss = train(train_loader, model, criterion, optimizer,device)
        val_loss = val(val_loader, model, criterion,device)
        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
            val_predict_label[:, 44:55].argmax(1),
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))
        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            # print('Find better model in Epoch {0}, savin
