import torch
import torch.nn as nn
from torchinfo import summary

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()

        self.fc_list = nn.ModuleList([nn.Linear(5,5) for i in range(3)])

    def forward(self, x):
        tmp = x
        for i in range(3):
            tmp = self.fc_list[i](tmp)

        return tmp

class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()

        self.fc1 = nn.Linear(5,5)
        self.fc2 = nn.Linear(5,5)
        self.fc3 = nn.Linear(5,5)

    def forard(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

testm1 = model1()
testm2 = model2()

summary(testm1)
summary(testm2)

# 반복문을 쓰더라도 둘 다 전체 파라미터 수 똑같이 잘 나옴
# 그리고 기본적으로 bias를 포함하네
# 또한 파라미터수는 batch수를 포함하지 않고 오직 한 데이터에 대한 파라미터 수이므로
# 전체 계산되는 파라미터 수 같은 경우는 batch가 추가 되더라도 trainable param의 수는 동일하지만
# 연산량은 batch_size의 배수 만큼 늘어나는 것임
# 뭐 어차피 test할 때는 보통 1개로만 inference하니까 머