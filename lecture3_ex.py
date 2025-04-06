import numpy as np # 행렬연산
import matplotlib.pyplot as plt # 시각화

import torch # 파이토치
import torch.nn as nn # 파이토치 모듈
import torch.nn.init as init # 초기화 관련 모듈 
import torch.optim as optim #최적화함수
from torch.utils.data import Dataset, DataLoader, random_split # 데이터셋을 학습에 용이하게 바꿈
import torch.nn.functional as F # 자주 쓰는 함수를 F로 따로 가져옴

import argparse
import wandb



parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=300)
parser.add_argument("--learningRate", type=float, default=0.01)
parser.add_argument("--epoch", type=int, default= 10)

args = parser.parse_args()

batch_size = args.batchSize
learning_rate = args.learningRate
num_epoch = args.epoch

rawDataset = "log.csv"

wandb.init(project="mlp", config=vars(args), name=f"bs{batch_size}_lr{learning_rate}_ep{num_epoch}")

#데이터셋 불러오기
class CSVDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=np.float32)
        self.x = torch.tensor(data[:, :-1])  # 특성 데이터
        self.y = torch.tensor(data[:, -1])   # 타겟 데이터

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = CSVDataset(rawDataset)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # layer 생성
        self.fc1 = nn.Linear(5, 25) # ABCDE가 입력되니까 5
        self.fc2 = nn.Linear(25, 125)
        self.fc3 = nn.Linear(125, 25)
        self.fc4 = nn.Linear(25, 1)
        
        self.dropout = nn.Dropout(0.5) # 연산마다 50% 비율로 랜덤하게 노드 삭제... 나중에
        
        self.batch_norm1 = nn.BatchNorm1d(25) # 1dimension이기 때문에 BatchNorm1d를 사용함.
        self.batch_norm2 = nn.BatchNorm1d(125)
        self.batch_norm3 = nn.BatchNorm1d(25)
    
    
    def forward(self, x): # 모델의 연산 순서를 정의
    # 1st layer
        x = x.view(-1, 5)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x) # activation function
        x = self.dropout(x)
    # 2nd layer
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
    # 3rd layer
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
    # 4th layer
        x = self.fc4(x)
        return x
    

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)

model= MLP()
model.apply(weight_init)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.3)
loss_fn = nn.MSELoss()


def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (Data, Score) in enumerate(train_loader): # A,B,C,D,E, Score로 이루어진 train set
        optimizer.zero_grad() # 최적화 초기화
        output = model(Data) #  모델에 input을 넣어서 예측값을 구한다
        loss = loss_fn(output, Score) # 손실함수, error 계산
        loss.backward() # 손실 함수 기준으로 역전파 설정
        optimizer.step() # 역전파 진행 및 가중치 업데이트

def evaluate(model, Test):
    model.eval()
    loss = 0

    if Test:
        with torch.no_grad(): # 모델의 평가 단계이므로 gradient가 업데이트 되지 않도록 함
            for data, score in test_loader:
                output = model(data)
                loss += loss_fn(output, score).item() # loss 누적
                loss += torch.sqrt(loss_fn(output, score))
            loss /= len(test_loader.dataset) 
    else:
        with torch.no_grad(): # 모델의 평가 단계이므로 gradient가 업데이트 되지 않도록 함
            for data, score in train_loader:
                output = model(data)
                loss += loss_fn(output, score).item() # loss 누적
                loss += torch.sqrt(loss_fn(output, score))
            loss /= len(train_loader.dataset)
    return loss

# 결과를 저장할 리스트
epoch_list = []
test_loss_list = []
train_loss_list = []

wandb.init(project="mlp", config=vars(args))

for epoch in range(1, num_epoch + 1):
    train(model, train_loader, optimizer)
    test_loss = evaluate(model, True)
    train_loss = evaluate(model, False)
    
    epoch_list.append(epoch)
    test_loss_list.append(test_loss)
    train_loss_list.append(train_loss)

    print("[EPOCH: {}], \tTest Loss(loss function): {:.4f}, \t Train loss(RSME): {:.6f} ".format(
        epoch, test_loss, train_loss
    ))

wandb.log({"avg_train_loss": train_loss/num_epoch})
