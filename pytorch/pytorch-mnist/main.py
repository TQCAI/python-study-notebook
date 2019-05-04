#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader


# In[2]:


train_data = torchvision.datasets.MNIST(
    './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
)


# In[3]:


test_data = torchvision.datasets.MNIST(
    './mnist', train=False, transform=torchvision.transforms.ToTensor()
)


batch_size=100
train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size)


# In[11]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


# In[12]:


model = Net()


# In[14]:


optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()


# In[15]:


for epoch in range(10):
    print(f'epoch {epoch + 1}')
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    cnt=0
    for batch_x, batch_y in train_loader:
        print(cnt)
        cnt+=1
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss = loss.data
        pred = torch.max(out, 1)[1]
        print(pred)
        train_correct = (pred == batch_y).sum()
        train_acc = train_correct.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Train Loss: {train_loss / batch_size:.6f}, Acc: {train_acc.type(torch.double) / batch_size:.6f}')


    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data
    print(f'Test Loss: {eval_loss / len(test_data):.6f}, Acc: {eval_acc / len(test_data):.6f}')


# In[ ]:




