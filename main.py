from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import data
import matplotlib.pyplot as plt
from skimage import io
import copy
import torch.optim as optim
current_path = os.getcwd()
dataset_name = 'mnist'
file_format = '.pkl'


class my_mnist(Dataset):
    """define a new dataset classes for the given mnist data"""
    def __init__(self, data, transform = None, data_flag = 'train'):
        """
        Args:
            data(np.nddary): we get it from the data.Get_data
            transform(callabke,optional): Optimal transforms to be applied on a sample
        """
        self.flag = data_flag
        self.transform = transform
        self.data = data[0]
        self.label = data[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        mnist_data = self.data[index]
        mnist_label = self.label[index]
        sample = {'data': mnist_data, 'label': mnist_label}
        #if self.transform:
        #    sample = self.transform(sample)
        return sample
        

# loding the train, val, test data
data_path = os.path.join(current_path, dataset_name) + file_format
train, val, test = data.Get_data(data_path)
mnist_train  = my_mnist(data = train, data_flag = 'train')
mnist_val = my_mnist(data = val, data_flag = 'val')
mnist_test = my_mnist(data = test, data_flag = 'test')


"""
uncomment the follow code if you would like to visualize the input data
"""
#fig = plt.figure()
#for i in range(len(mnist_train)):
#    sample = mnist_train[i]
#    sample['data'].resize((28,28))
#    plt.imshow(sample['data'])
#    plt.show()
#    if i == 3:
#        break



data_set = {'train': train, 'val':val, 'test':test}
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
    ]
    ),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
    ]
    ),
    
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
    ]
    ),
}
image_datasets = {x: my_mnist(data = data_set[x], transform = data_transforms[x], data_flag = x)
                 for x in ['train', 'val', 'test']}
dataset_sizes = {x : len(image_datasets[x]) for x in ['train', 'val', 'test']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 128, shuffle = True, num_workers = 4)
              for x in ['train', 'val', 'test']}

use_gpu = torch.cuda.is_available()


# In[74]:


class MNIST_Net(nn.Module):
    def __init__(self, mode):
        super(MNIST_Net,self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 784)
        if mode == 'autoencoder':
            self.mode = 'autoencoder'
        else:
            self.mode = 'classification'
            self.fc5 = nn.Linear(784, 10)
        
    def forward(self, inputs):
        x = inputs.view(inputs.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        if self.mode == 'autoencoder':
            return x
        elif self.mode == 'classification':
            x = self.fc5(x)
            return x
        
net_ae = MNIST_Net('autoencoder')
net_cls = MNIST_Net('classification')
net_pre = MNIST_Net('classification')
net_ae = net_ae.cuda()
net_cls = net_cls.cuda()
net_pre = net_pre.cuda()


#define the loss function and optimizer

# use the mse_loss if  used as an autoencoder
criterion_ae = nn.MSELoss()

# use the CrossEntropyLoss if used as a classifier
criterion_cls = nn.CrossEntropyLoss()


optimizer_ae = optim.SGD(net_ae.parameters(), lr = 0.001, momentum = 0.9)


# Decay LR by a factor of 0.1 every 10 epoch
exp_lr_scheduler_ae = lr_scheduler.StepLR(optimizer_ae, step_size = 20, gamma = 0.1)


# In[76]:


def train(net, criterion, optimizer, scheduler, mode,num_epoches = 100):
    print('mode:{}'.format(mode))
    train_loss_record = []
    val_loss_record = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches-1))
        print('-'* 20)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
            running_loss = 0.0
            running_corrects = 0
            
            for datasets in dataloaders[phase]:
                inputs = datasets['data']
                if mode == 'autoencoder':
                    labels = datasets['data']
                elif mode == 'cls':
                    labels = datasets['label']
        
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(lables)
                optimizer.zero_grad()
                
                #forward
                outputs = net(inputs)
                
                if mode == 'cls':
                    _, preds = torch.max(outputs.data, 1)
                
                # uncomment this if used as an autoencoder
                loss = criterion(outputs, labels)                
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.data[0] * inputs.size(0)
                
                if mode == 'cls':
                    running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                train_loss_record.append(epoch_loss)
                
            elif phase == 'val':
                val_loss_record.append(epoch_loss)
                
            if mode == 'cls':
                epoch_acc = running_corrects / dataset_sizes[phase]
                if phase == 'train':
                    train_acc.append(epoch_acc)
                elif phase == 'val':
                    val_acc.append(epoch_acc)
                print('{} Acc:{:.4f}'.format(phase, epoch_acc))
                
            print('{} Loss:{:.4f}'.format(phase, epoch_loss))
    print('Training Finished')
    if mode == 'autoencoder':
        print('abc')
        ae_model = copy.deepcopy(net.state_dict())
        return train_loss_record, val_loss_record, ae_model
    else:
        return train_loss_record, val_loss_record, train_acc, val_acc



def show_fig(x, y, x_label, y_label, save_name):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y, x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_name)
    plt.show()




y = range(1)
train_loss, val_loss, model_param_ae = train(net_ae,criterion_ae, optimizer_ae, exp_lr_scheduler_ae, mode = 'autoencoder',num_epoches = 1)
show_fig(train_loss, y, 'num_epoch', 'ae_train_loss', 'autoencoder_train_loss.png')
show_fig(val_loss, y, 'num_epoch', 'ae_val_loss','autoencoder_train_loss.png')
print(type(model_param_ae))



optimizer_cls = optim.SGD(net_cls.parameters(), lr = 0.001, momentum = 0.9)
exp_lr_scheduler_cls = lr_scheduler.StepLR(optimizer_cls, step_size = 1, gamma = 0.1)
train_loss_cls, val_loss_cls, train_acc, val_acc = train(net_cls, criterion_cls, optimizer_cls, exp_lr_scheduler_cls, mode ='cls', num_epoches = 1)
label_y = range(1)
show_fig(train_loss_cls, label_y, 'num_epoch', 'cls_train_loss', 'classification_train_loss.png')
show_fig(val_loss_cls, label_y, 'num_epoch', 'cls_val_loss', 'classification_val_loss.png')
show_fig(train_acc, label_y, 'num_epoch','cls_train_acc', 'classification_train_acc.png')
show_fig(val_acc, label_y, 'num_epoch', 'cls_val_acc', 'classification_val_acc.png')




net_pre = MNIST_Net("classification")
net_pre = net_pre.cuda()
net_pre.load_state_dict(model_param_ae)
optimizer_pre = optim.SGD(net_pre.parameters(), lr = 0.001, momentum = 0.9)
exp_lr_scheduler_pre = lr_scheduler.StepLR(optimizer_pre, step_size = 40, gamma = 0.1)
train_loss_cls_pre, val_loss_cls_pre, train_acc_pre, val_acc_pre = train(net_pre, criterion_cls, optimizer_pre,                                                                  exp_lr_scheduler_pre, mode = 'cls',                                                                  num_epoches = 1
                                                                )
label_y  = range(1)
show_fig(train_loss_cls_pre, label_y, 'num_epoch', 'cls_train_loss_pre', 'pre_classification_train_loss.png')
show_fig(val_loss_cls_pre, label_y, 'num_epoch', 'cls_val_loss_pre', 'pre_classification_val_loss.png')
show_fig(train_acc_pre, label_y, 'num_epoch', 'cls_train_acc_pre', 'pre_classification_train_acc.png')
show_fig(val_acc_pre, label_y, 'num_epoch', 'cls_val_acc_pre', 'pre_classification_val_acc.png')



