"""
Sort-of-CLEVRを忠実に再現したモデル
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#モデルの訓練とテスト，重みの保存を定義(main.pyで主に使う)
class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad() #全ての変数の勾配を初期化
        output = self(input_img, input_qst)
        loss = F.cross_entropy(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct = pred.eq(label.data).cpu().sum()
        return correct/len(label)*100.

    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        return correct/len(label)*100.

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))

"""
以下，モデルの定義
1. ConvInputModel:  画像処理用のCNN
2. FCOutputModel:   RNのfの2層目，3層目を定義
3. RN:              CNNの最終出力の読み込みからfによるanswer_vecの出力までを実装，BasicModelを継承
4. CNN_MLP:         CNN+RNとの比較対象，BasicModelを継承
"""
class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)


    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return F.log_softmax(x)


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')

        self.conv = ConvInputModel()

        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((256+2)*2+11, 2000) #左の数字はL.161, L.164と対応

        self.g_fc2 = nn.Linear(2000, 2000)
        self.g_fc3 = nn.Linear(2000, 2000)
        self.g_fc4 = nn.Linear(2000, 2000)

        self.f_fc1 = nn.Linear(2000, 1000)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]

        self.coord_tensor = Variable(torch.FloatTensor(args.batch_size, 25, 2))
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 256 x 5 x 5)

        #g
        mb = x.size()[0] #64
        n_channels = x.size()[1] #256
        d = x.size()[2] #5
        x_flat = x.view(mb, n_channels, d*d).permute(0, 2, 1) #(64 x 256 x 5 x 5) -> (64 x 25 x 256)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2) #(64x25x(256 + 2)), 座標テンソルを列方向に結合している

        # add question everywhere
        qst = torch.unsqueeze(qst, 1) #(64x1x11) tensor 次元が1の方向に1上がったテンソルができる
        qst = qst.repeat(1, 25, 1) #同じ(64x25x11)tensorが下に25個できる np.tileみたいな感じ
        qst = torch.unsqueeze(qst, 2) #(64x25x1x11)

        # cast all pairs against each other
        """
        一つのオブジェクト = (o_i, o_j, qst)
        x_i <- o_i (論文中)
        x_j <- o_j + qst
        """
        x_i = torch.unsqueeze(x_flat, 1) # (64x1x25x258)
        x_i = x_i.repeat(1, 25, 1, 1) # (64x25x25x26)

        x_j = torch.unsqueeze(x_flat, 2) # (64x25x1x258)
        x_j = torch.cat([x_j, qst], 3) # (64x25x1x(258+11))
        x_j = x_j.repeat(1, 1, 25, 1) # (64x25x25x(258+11))

        # concatenate all together
        #オブジェクトをd*dの数だけ作ったので，RNの入力用にすべて結合
        x_full = torch.cat([x_i, x_j], 3) # (64x25x25x(258+258+11))

        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d, 527) #RNの最初のNNに入力するため形を変形
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_) #この段階での形は，(mb(=64) x d x d x d x d x 2000)

        #reshape again and sum
        x_g = x_.view(mb, d*d*d*d, 2000)
        x_g = x_g.sum(1).squeeze() #element-wise sum

        #f
        x_f = self.f_fc1(x_g) #fの1層目
        x_f = F.relu(x_f)

        return self.fcout(x_f) #fの2層目以降


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv = ConvInputModel()
        self.fc1 = nn.Linear(5*5*256 + 11, 1000)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.conv(img) # x = (64 x 256 x 5 x 5)

        #fully connected layers
        x = x.view(x.size(0), -1)
        x_ = torch.cat((x, qst), 1)  # Concat question
        x_ = self.fc1(x_)
        x_ = F.relu(x_)

        return self.fcout(x_)
