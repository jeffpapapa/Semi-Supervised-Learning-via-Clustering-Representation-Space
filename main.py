import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os
from MCMC_loss import DBI, margin
import random
import torchvision

batch_size = 50 ### default : 50
eval_batch_size = 100
unlabeled_batch_size = 300 #default:300
num_iter_per_epoch = 200
iter_eval_freq = 20 #iter freq
eval_freq = 1 #epoch freq
lr = 0.001
cuda_device = "0"

output_dim = 10
cluster_dim = 300


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | coil100')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=10) ###default=120
parser.add_argument('--epoch_decay_start', type=int, default=3) ###default=80
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--method', default='mcmc')
parser.add_argument('--num_label_per_class', type=int, default=10)


opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

cuda_gpu = torch.cuda.is_available()

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def train(model, x, y, ul_x, optimizer):

    ce = nn.CrossEntropyLoss()
    y_pred, z_l = model(x)
    ce_loss = ce(y_pred, y)
    
    y_pred = y_pred.type(torch.uint8)
    
        
#    loss_DB = DBI(10, 128, cuda_gpu)
#    label_DB_loss = loss_DB(z_l, y).cuda()
    
    loss = ce_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#    del loss_DB

    return ce_loss

def train_mcmc(model, x, y, ul_x, optimizer):

    ce = nn.CrossEntropyLoss()
    y_pred, z_l = model(x)
    ce_loss = ce(y_pred, y)
    
    y_pred = y_pred.type(torch.uint8)
    
    ul_y, z_ul = model(ul_x)

        
    loss_DB = DBI(output_dim, cluster_dim, cuda_gpu)
    loss_ML = margin(output_dim,10, cuda_gpu)


    label_DB_loss = loss_DB(z_l, y).cuda()
        
        
#        print(output_u_label)
#        print('\n=====================分隔線============================\n')
#        print(target_var)
#        print(z.size())
#        print(z[sl[0]:].size())
    
    _, pre_ul_y = torch.max(ul_y, 1)
    
    #ul_y = ul_y.type(torch.uint8)
    total_DB_loss = loss_DB(z_ul, pre_ul_y).cuda()
    
    z_ul = z_ul.type(torch.float)
    
    margin_loss = loss_ML(z_ul, pre_ul_y).cuda()

    loss = ce_loss+ label_DB_loss + total_DB_loss + margin_loss
    


    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    del loss_DB, loss_ML

    return ce_loss, label_DB_loss, total_DB_loss, margin_loss


def eval(model, x, y):

    y_pred, z = model(x)
    prob, idx = torch.max(y_pred, dim=1)
    return torch.eq(idx, y).float().mean()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        
if opt.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=opt.dataroot, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=eval_batch_size, shuffle=True)
    
elif opt.dataset == 'coil100':
    num_labeled = opt.num_label_per_class
    data_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    
    train_data = torchvision.datasets.ImageFolder(
        root=opt.dataroot,
        transform= data_transforms
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=72,
        num_workers=1,
        shuffle=False
    )

else:
    raise NotImplementedError

train_data = []
train_target = []

for (data, target) in train_loader:
    train_data.append(data)
    train_target.append(target)

train_data = torch.cat(train_data, dim=0)
train_target = torch.cat(train_target, dim=0)

#valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
#valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]


if opt.dataset == 'mnist':
    ### for our experiments, we need to random split labeled/unlabeled data
    
    single_label_num = int(opt.num_label_per_class)
    print('num of label per class : ',single_label_num)
    label_data_idx = []
    unlabel_data_idx = []
    for num in range(0,10):
        num_idx = [i for i,v in enumerate(train_target) if v==num]
        random.shuffle(num_idx)
        label_data_idx = label_data_idx + num_idx[:single_label_num]
        unlabel_data_idx = unlabel_data_idx + num_idx[single_label_num:]
    
    random.shuffle(label_data_idx)
    random.shuffle(unlabel_data_idx)

    labeled_train = train_data[label_data_idx]
    labeled_target = train_target[label_data_idx]
    test_data = train_data[unlabel_data_idx]
    test_target = train_target[unlabel_data_idx]
    
elif opt.dataset == 'coil100':
    single_label_num = int(opt.num_label_per_class)
    print('num of label per class : ',single_label_num)
    label_data_idx = []
    unlabel_data_idx = []
    for num in range(0,100):
        num_idx = [i for i,v in enumerate(train_target) if v==num]
        random.shuffle(num_idx)
        label_data_idx = label_data_idx + num_idx[:single_label_num]
        unlabel_data_idx = unlabel_data_idx + num_idx[single_label_num:]
    
    random.shuffle(label_data_idx)
    random.shuffle(unlabel_data_idx)

    labeled_train = train_data[label_data_idx]
    labeled_target = train_target[label_data_idx]
    test_data = train_data[unlabel_data_idx]
    test_target = train_target[unlabel_data_idx]
else:
    ### set your labeled data/target here
    labeled_train, labeled_target = train_data[:num_labeled, ], train_target[:num_labeled, ]
    
### set your unlabeled data here  
### in my exp., unlabeled data includes "all training data (label+unlabel)"
unlabeled_train = train_data[:, ]

print('labeled data amount : ', len(labeled_train))
print('training(labeled+unlabeled) data amount : ', len(unlabeled_train))

model = tocuda(Net(output_dim, cluster_dim))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)
#optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.1)

acc_record = []

# train the network
for epoch in range(opt.num_epochs):
    
    if epoch > opt.epoch_decay_start:
        decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
        optimizer.lr = decayed_lr
        print('new lr : ', decayed_lr)
        optimizer.betas = (0.9, 0.99)

    for i in range(num_iter_per_epoch):

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        batch_indices_unlabeled = torch.LongTensor(np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
        ul_x = unlabeled_train[batch_indices_unlabeled]

#        v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
#                                optimizer)
        
        if(opt.method=='su'):
            loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                    optimizer)
    
    #        if i % 100 == 0:
    #            print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.item(), "CE Loss :", ce_loss.item())
                
            if i % iter_eval_freq == 0:
                print("Epoch :", epoch, "Iter :", i, "Loss :", loss.item())
    #            print('entropy loss : ',en_loss.item())
        
        
        else:
            ce_loss, label_DB_loss, total_DB_loss, margin_loss = train_mcmc(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                    optimizer)
    
    #        if i % 100 == 0:
    #            print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.item(), "CE Loss :", ce_loss.item())
                
            if i % iter_eval_freq == 0:
                print("Epoch :", epoch, "Iter :", i, "CE Loss :", ce_loss.item(), "LDB Loss :", label_DB_loss.item(),
                      "TDB Loss :", total_DB_loss.item(),"MM Loss :", margin_loss.item())
    #            print('entropy loss : ',en_loss.item())
            
            

    if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:
        
        test_batch_size = 100
        batch = int(len(train_data)/test_batch_size)
        if(test_batch_size *batch != len(train_data)):
            batch+=1
        train_accuracy = 0.0
        counter = 0
        for test_batch_idx in range(batch):
            data = train_data[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size]
            target = train_target[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size]
            n = data.size()[0]
            acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
            train_accuracy += n*acc
            counter += n
        print('train data amount:',counter)
        print("Train accuracy :", train_accuracy.item()/counter)
        
        test_accuracy = 0.0
        counter = 0        
        test_batch_size = 100
        batch = int(len(test_data)/test_batch_size)
        if(test_batch_size *batch != len(test_data)):
            batch+=1
        train_accuracy = 0.0
        counter = 0
        for test_batch_idx in range(batch):
            data = test_data[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size]
            target = test_target[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size]
            n = data.size()[0]
            acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
            test_accuracy += n*acc
            counter += n
        print('test(unlabeled) data amount:',counter)
        print("test(unlabeled) accuracy :", test_accuracy.item()/counter)
        acc_record.append(test_accuracy.item()/counter)
        

test_accuracy = 0.0
counter = 0        
test_batch_size = 100
batch = int(len(test_data)/test_batch_size)
if(test_batch_size *batch != len(test_data)):
    batch+=1
train_accuracy = 0.0
counter = 0
for test_batch_idx in range(batch):
    data = test_data[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size]
    target = test_target[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size]
    n = data.size()[0]
    acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    test_accuracy += n*acc
    counter += n

print("Full test(unlabeled) accuracy :", test_accuracy.item()/counter)

import json 
with open(str(opt.method)+'+'+str(opt.dataset)+'_acc_record_'+str(opt.num_label_per_class)+'.json', 'w') as outfile:
    json.dump(acc_record, outfile)
