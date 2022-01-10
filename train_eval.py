# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(model, train_iter, dev_iter, config, self_speed_flag):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=float(config.weight_decay))



    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step, gamma=0.1)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    class_weights = [1, 1, 1]
    w = torch.FloatTensor(class_weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    f1_side_action_list = []
    max_vali_f1_score = 0
    for epoch in range(config.num_epochs):
        model.train()
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (self_speed_info, all_objects_info, action_label, single_csv_file, three_kinds_of_object_num_in_an_img) in enumerate(train_iter):
            outputs, _, _ = model(self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img, self_speed_flag)
            # print(len(all_objects_info))
            # print(all_objects_info[0].shape)
            # print(all_objects_info[1].shape)
            # print(all_objects_info[2].shape)
            # print(trains[0].shape, trains[1].shape)
            # exit()
            model.zero_grad()
            loss = criterion(outputs, action_label)
            # loss = F.cross_entropy(outputs, action_label)
            # print(loss.item(), total_batch)

            predict_action = torch.sigmoid(outputs) > 0.5
            # print(pred_reason[0])
            # print(pred_action[0])
            # print(reasonBatch[0])
            # print(predict_reason[0])
            # exit()
            # f1_side_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')
            f1_side_action = f1_score(action_label.cpu().data.numpy(), predict_action.cpu().data.numpy(), average='samples')
            f1_side_action_list.append(f1_side_action)
            # print(loss.item(), f1_side_action, total_batch)
            loss.backward()
            optimizer.step()

            # if total_batch % 100 == 0:
            #     # 每多少轮输出在训练集和验证集上的效果
            #     true = labels.data.cpu()
            #     predic = torch.max(outputs.data, 1)[1].cpu()
            #     train_acc = metrics.accuracy_score(true, predic)
            #     dev_acc, dev_loss = evaluate(config, model, dev_iter)
            #     if dev_loss < dev_best_loss:
            #         dev_best_loss = dev_loss
            #         torch.save(model.state_dict(), config.save_path)
            #         improve = '*'
            #         last_improve = total_batch
            #     else:
            #         improve = ''
            #     time_dif = get_time_dif(start_time)
            #     msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
            #     print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
            #     # writer.add_scalar("loss/train", loss.item(), total_batch)
            #     # writer.add_scalar("loss/dev", dev_loss, total_batch)
            #     # writer.add_scalar("acc/train", train_acc, total_batch)
            #     # writer.add_scalar("acc/dev", dev_acc, total_batch)
            #     model.train()
            total_batch += 1
            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过1000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
        train_f1_score_per_epoch = np.mean(f1_side_action_list)
        vali_f1_score_per_epoch = evaluate(model, dev_iter, self_speed_flag)
        save_flag = (train_f1_score_per_epoch + vali_f1_score_per_epoch) / 2
        if max_vali_f1_score < save_flag:
            max_vali_f1_score = save_flag
            max_train_avergae = train_f1_score_per_epoch
            max_vali_avergae = vali_f1_score_per_epoch
            torch.save(model.state_dict(), config.save_path + '.ckpt')
            # torch.save(model.state_dict(), config.save_path + str(max_vali_f1_score)[0:6] + '.ckpt')
        torch.save(model.state_dict(), config.save_path + "_final" + '.ckpt')
        print("Train F1-score:", train_f1_score_per_epoch, "Validation F1-score:", vali_f1_score_per_epoch, max_train_avergae, max_vali_avergae)
        # scheduler.step()
        # if flag:
        #     break
    # writer.close()
    # test(config, model, test_iter)


# def test(config, model, test_iter):
#     # test
#     model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#     start_time = time.time()
#     test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
#     msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
#     print(msg.format(test_loss, test_acc))
#     print("Precision, Recall and F1-Score...")
#     print(test_report)
#     print("Confusion Matrix...")
#     print(test_confusion)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)


def evaluate(model, data_iter, self_speed_flag):
    model.eval()
    f1_side_action_list = []
    with torch.no_grad():
        for i, (self_speed_info, all_objects_info, action_label, _, three_kinds_of_object_num_in_an_img) in enumerate(data_iter):
            outputs,_, _ = model(self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img, self_speed_flag)
            # loss = criterion(outputs, action_label)
            # print(loss.item(), total_batch)

            predict_action = torch.sigmoid(outputs) > 0.5

            # f1_side_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')
            f1_side_action = f1_score(action_label.cpu().data.numpy(), predict_action.cpu().data.numpy(), average='samples')
            f1_side_action_list.append(f1_side_action)
            # print(loss.item(), f1_side_action, total_batch)
    f1_side_action_epoch_mean = np.mean(f1_side_action_list)
    return f1_side_action_epoch_mean