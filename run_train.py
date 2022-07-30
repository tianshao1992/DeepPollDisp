import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import time as record
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from visual_data import matplotlib_vision
import basic_model, read_data
from basic_model import cal_fields_error


train_size = 0.8
path = "results_" + str(train_size) + "\\"
save_path = path + "train\\"
if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 数据读取
names, label = read_data.data_gen(train_size=train_size)

# 数据归一化
label_norm = read_data.data_norm(label[0])
field_norm = read_data.data_norm(np.array([[0,], [255,]]))

# torch.save({'design_norm': design_norm, 'fields_norm': fields_norm, 'target_norm': target_norm},
#            'data/var_norm.pth')


train_dataset = read_data.custom_dataset(names[0], label[0])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)

valid_dataset = read_data.custom_dataset(names[1], label[1])
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, drop_last=False)

# 模型建立
ge = basic_model.generator(nz=4, num_layers=8, num_filters=32, num_convs=4, outputs=(1, 512, 512)).cuda()

# 损失及优化方式
fields_loss = nn.MSELoss().cuda()

bound_epoch = [5001, ]
optimizer = torch.optim.Adamax([{'params': ge.parameters(), 'lr': 0.001},])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=bound_epoch, gamma=0.1)


visual = matplotlib_vision(save_path)

train_loss_log, train_error_log, train_target_log = [], [], []
valid_loss_log, valid_error_log, valid_target_log = [], [], []


start_time = record.time()
for epoch in range(1, bound_epoch[-1]):
    # 训练
    for it in range(int(len(names[0])/16)):
        ge.train()
        f_train, d_train = next(iter(train_loader))
        d_train = label_norm.norm(d_train).cuda()
        f_train = f_train.cuda()
        f_train_ = ge(d_train)

        train_fields_loss = fields_loss(f_train_, f_train)
        total_loss = train_fields_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (it+1) % 500 == 0:
            # 验证
            f_valid, d_valid = next(iter(valid_loader))
            d_valid = label_norm.norm(d_valid).cuda()
            f_valid = f_valid.cuda()

            ge.eval()
            with torch.no_grad():
                f_train_ = ge(d_train)
                train_fields_loss = fields_loss(f_train_, f_train)
                f_valid_ = ge(d_valid)
                valid_fields_loss = fields_loss(f_valid_, f_valid)

            train_error_log.append(cal_fields_error(f_train, f_train_))
            valid_error_log.append(cal_fields_error(f_valid, f_valid_))

            train_loss_log.append([train_fields_loss.item(), ])
            valid_loss_log.append([valid_fields_loss.item(), ])

            elapsed = record.time() - start_time

            print(
                'epoch: %d / %d, Cost: %.2f, Lr: %.2e , '
                'train fields Loss = %.2e, test fields Loss = %.2e , '
                % (epoch, bound_epoch[-1], elapsed, optimizer.state_dict()['param_groups'][0]['lr'],
                   train_loss_log[-1][0], valid_loss_log[-1][0],
                   )
            )
            start_time = record.time()

    # 学习率调整
    scheduler.step()

    if epoch % 10 == 0:
        # 模型保存
        torch.save({'epoch': epoch, 'ge': ge.state_dict()}, os.path.join(path, 'latest_model.pth'))
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}

        sio.savemat(save_path + 'train_process.mat', {'train_error': np.array(train_error_log),
                                                  'valid_error': np.array(valid_error_log),
                                                  'train_loss': np.array(train_loss_log),
                                                  'valid_loss': np.array(valid_loss_log),})

        # 结果统计： 损失函数
        plt.figure(100, figsize=(30, 15))
        plt.clf()
        plt.subplot(211)
        visual.plot_loss(np.arange(1, len(train_loss_log)+1), np.array(train_loss_log)[:, 0], label='train_f_loss')
        visual.plot_loss(np.arange(1, len(train_loss_log)+1), np.array(valid_loss_log)[:, 0], label='valid_f_loss')

        plt.savefig('log_loss.svg')

        c_valid = c_valid.numpy().transpose((0, 2, 3, 1))
        f_valid, f_valid_ = f_valid.cpu().numpy().transpose((0, 2, 3, 1)), f_valid_.detach().cpu().numpy().transpose((0, 2, 3, 1))

        c_train = c_train.numpy().transpose((0, 2, 3, 1))
        f_train, f_train_ = f_train.cpu().numpy().transpose((0, 2, 3, 1)), f_train_.detach().cpu().numpy().transpose((0, 2, 3, 1))


        # 结果统计： 选择一个压力分布和温度分布绘图
        plt.figure(101, figsize=(30, 15))
        plt.clf()
        visual.plot_fields_2d(f_train[0, :, :, :], f_train_[0, :, :, :])
        plt.savefig(save_path + 'fields_train.jpg')

        plt.figure(101, figsize=(30, 15))
        plt.clf()
        visual.plot_fields_2d(f_valid[0, :, :, :], f_valid_[0, :, :, :])
        plt.savefig(save_path + 'fields_valid.jpg')