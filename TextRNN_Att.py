
# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextRNN_Att'
        self.weight_decay = 1e-9
        self.decay_step = 200
                                      # 预训练词向量
        self.save_path = "trained_model/" + self.model_name
        # print(self.embedding_pretrained.size()) # random: None # pre_trained-train: torch.Size([4762, 300]) 我认为这是 由 4762 个 onehot的词向量 变成了 300 个维度更加低的矢量
        # exit()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练

        self.self_speed_info_size = 2
        self.object_info_size = 7
        self.lane_info_size = 2
        self.traffic_info_size = 5

        self.self_speed_middle_info_size = 10
        self.object_middle_info_size = 50
        self.lane_middle_info_size = 100
        self.traffic_middle_info_size = 10

        self.num_classes = 3                                            # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 1000                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        # print(self.embed) # pre_trained : 300 # random: 300 当我们选择用词向量的时候，我们就不会再使用这里的字向量
        # exit()
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.self_speed_fc = nn.Linear(config.self_speed_info_size, config.self_speed_middle_info_size)

        # moveable_object_info_lstm
        self.moveable_object_lstm = nn.LSTM(config.object_info_size, config.hidden_size, config.num_layers,
                                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh_moveable_object = nn.Tanh()
        self.moveable_object_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.moveable_object_fc = nn.Linear(config.hidden_size * 2, config.object_middle_info_size)


        # lane_info_lstm
        self.lane_info_lstm = nn.LSTM(config.lane_info_size, config.hidden_size, config.num_layers,
                                      bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh_lane = nn.Tanh()
        self.lane_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.lane_info_fc = nn.Linear(config.hidden_size * 2, config.lane_middle_info_size)


        # traffic_info_lstm
        self.traffic_info_lstm = nn.LSTM(config.traffic_info_size, config.hidden_size, config.num_layers,
                                         bidirectional=True, batch_first=True, dropout=config.dropout)
        self.traffic_light_object = nn.Tanh()
        self.traffic_light_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.traffic_info_fc = nn.Linear(config.hidden_size * 2, config.traffic_middle_info_size)


        # over_all_info_fc
        self.middle_fc_feature1 = 20
        self.middle_fc_feature2 = 10
        self.middle_fc_feature0 = 5

        self.drop = nn.Dropout(p=0.5)
        self.relu1 = nn.ReLU(inplace=True)

        over_all_input_size = config.self_speed_middle_info_size + config.object_middle_info_size + config.lane_middle_info_size + config.traffic_middle_info_size

        self.over_all_fc = nn.Linear(over_all_input_size, self.middle_fc_feature1)
        self.middle_fc1 = nn.Linear(self.middle_fc_feature1, self.middle_fc_feature2)
        self.middle_fc2 = nn.Linear(self.middle_fc_feature2, config.num_classes)


    def forward(self, self_speed_info, all_objects_info):

        self_speed_tensor_info_list = self_speed_info
        all_object_info_tuple = all_objects_info
        # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)

        moveable_object_lstm_out, _ = self.moveable_object_lstm(all_object_info_tuple[0])

        moveable_object_M = self.tanh_moveable_object(moveable_object_lstm_out)  # [128, 32, 256]
        # print(moveable_object_M.shape)
        # print(self.moveable_object_weight.shape)
        # exit()
        moveable_object_alpha = F.softmax(torch.matmul(moveable_object_M, self.moveable_object_weight), dim=1).unsqueeze(-1)  # [128, 32, 1]
        moveable_object_out = moveable_object_lstm_out * moveable_object_alpha  # [128, 32, 256]
        moveable_object_out = torch.sum(moveable_object_out, 1)  # [50, 256]
        moveable_object_out = F.relu(moveable_object_out)
        object_middle_info = self.moveable_object_fc(moveable_object_out)



        lane_lstm_out, _ = self.lane_info_lstm(all_object_info_tuple[1])

        lane_M = self.tanh_lane(lane_lstm_out)  # [128, 32, 256]

        lane_alpha = F.softmax(torch.matmul(lane_M, self.lane_weight), dim=1).unsqueeze(-1)  # [128, 32, 1]
        lane_out = lane_lstm_out * lane_alpha  # [128, 32, 256]
        lane_out = torch.sum(lane_out, 1)  # [50, 256]
        lane_out = F.relu(lane_out)
        lane_middle_info = self.lane_info_fc(lane_out)



        traffic_light_lstm_out, _ = self.traffic_info_lstm(all_object_info_tuple[2])

        traffic_light_M = self.tanh_moveable_object(traffic_light_lstm_out)  # [128, 32, 256]

        traffic_light_alpha = F.softmax(torch.matmul(traffic_light_M, self.traffic_light_weight), dim=1).unsqueeze(-1)  # [128, 32, 1]
        traffic_light_out = traffic_light_lstm_out * traffic_light_alpha  # [128, 32, 256]
        traffic_light_out = torch.sum(traffic_light_out, 1)  # [50, 256]
        traffic_light_out = F.relu(traffic_light_out)
        traffic_light_middle_info = self.traffic_info_fc(traffic_light_out)

        over_all_middle_info = torch.cat(
            (self_speed_middle_info, object_middle_info, lane_middle_info, traffic_light_middle_info), dim=1)
        # print(self_speed_middle_info.shape, object_middle_info)
        # print(over_all_middle_info.shape)
        # exit()
        over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
        # print(over_all_middle_info.shape)
        middle_fc_feature = self.drop(self.relu1(self.over_all_fc(over_all_middle_info)))
        output = self.middle_fc1(middle_fc_feature)
        # output = self.middle_fc0(output)
        output = self.middle_fc2(output)

        return output

    def show_attention(self, self_speed_info, all_objects_info):

        self_speed_tensor_info_list = self_speed_info
        all_object_info_tuple = all_objects_info
        # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)

        moveable_object_lstm_out, _ = self.moveable_object_lstm(all_object_info_tuple[0])

        moveable_object_M = self.tanh_moveable_object(moveable_object_lstm_out)  # [128, 32, 256]
        moveable_object_alpha = F.softmax(torch.matmul(moveable_object_M, self.moveable_object_weight), dim=1).unsqueeze(-1)  # [128, 32, 1]
        moveable_object_out = moveable_object_lstm_out * moveable_object_alpha  # [128, 32, 256]
        moveable_object_out = torch.sum(moveable_object_out, 1)  # [50, 256]

        object_middle_info = self.moveable_object_fc(moveable_object_out)



        lane_lstm_out, _ = self.lane_info_lstm(all_object_info_tuple[1])

        lane_M = self.tanh_lane(lane_lstm_out)  # [128, 32, 256]
        lane_alpha = F.softmax(torch.matmul(lane_M, self.lane_weight), dim=1).unsqueeze(-1)  # [128, 32, 1]
        lane_out = lane_lstm_out * lane_alpha  # [128, 32, 256]
        lane_out = torch.sum(lane_out, 1)  # [50, 256]

        lane_middle_info = self.lane_info_fc(lane_out)



        traffic_light_lstm_out, _ = self.traffic_info_lstm(all_object_info_tuple[2])

        traffic_light_M = self.tanh_moveable_object(traffic_light_lstm_out)  # [128, 32, 256]
        traffic_light_alpha = F.softmax(torch.matmul(traffic_light_M, self.traffic_light_weight), dim=1).unsqueeze(-1)  # [128, 32, 1]
        traffic_light_out = traffic_light_lstm_out * traffic_light_alpha  # [128, 32, 256]
        traffic_light_out = torch.sum(traffic_light_out, 1)  # [50, 256]

        traffic_light_middle_info = self.traffic_info_fc(traffic_light_out)

        over_all_middle_info = torch.cat(
            (self_speed_middle_info, object_middle_info, lane_middle_info, traffic_light_middle_info), dim=1)
        # print(self_speed_middle_info.shape, object_middle_info)
        # print(over_all_middle_info.shape)
        # exit()
        over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
        # print(over_all_middle_info.shape)
        middle_fc_feature = self.drop(self.relu1(self.over_all_fc(over_all_middle_info)))
        output = self.middle_fc1(middle_fc_feature)
        # output = self.middle_fc0(output)
        output = self.middle_fc2(output)

        return output, moveable_object_alpha, lane_alpha, traffic_light_alpha


def save_a_attention_for_a_img(save_csv_path, one_img_predict_action, one_img_action_label, one_img_moveable_object_alpha, one_img_lane_alpha, one_img_traffic_light_alpha, moveable_object_info, lane_info, traffic_object_info):
    list = []

    one_img_predict_action = one_img_predict_action.cpu().numpy().tolist()
    one_img_action_label = one_img_action_label.cpu().numpy().tolist()

    one_img_moveable_object_alpha = one_img_moveable_object_alpha.cpu().numpy().tolist()
    one_img_lane_alpha = one_img_lane_alpha.cpu().numpy().tolist()
    one_img_traffic_light_alpha = one_img_traffic_light_alpha.cpu().numpy().tolist()

    moveable_object_info = moveable_object_info.cpu().numpy().tolist()
    lane_info = lane_info.cpu().numpy().tolist()
    traffic_object_info = traffic_object_info.cpu().numpy().tolist()


    with open(save_csv_path, "a", newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')            # write the result to the new csv
        writer.writerow(one_img_predict_action)
        writer.writerow(one_img_action_label)

        for i in range(len(one_img_moveable_object_alpha)):
            list.append(one_img_moveable_object_alpha[i])
            writer.writerow(list)
            list = []

        for i in range(len(moveable_object_info)):
            list.append(moveable_object_info[i])
            writer.writerow(list)
            list = []

        for i in range(len(one_img_lane_alpha)):
            list.append(one_img_lane_alpha[i])
            writer.writerow(list)
            list = []
        for i in range(len(lane_info)):
            list.append(lane_info[i])
            writer.writerow(list)
            list = []

        for i in range(len(one_img_traffic_light_alpha)):
            list.append(one_img_traffic_light_alpha[i])
            writer.writerow(list)
            list = []
        for i in range(len(traffic_object_info)):
            list.append(traffic_object_info[i])
            writer.writerow(list)
            list = []

def show_RNNattention(model_path, pseudo_img_name, data_iter, config, device, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum):
    model = Model(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    f1_side_action_list = []
    with torch.no_grad():
        for i, (self_speed_info, all_objects_info, action_label, img_name_list) in enumerate(data_iter):
            if pseudo_img_name in img_name_list:
                img_index = img_name_list.index(pseudo_img_name)
                # print(img_index)
                # exit()
                outputs, moveable_object_alpha, lane_alpha, traffic_light_alpha = model.show_attention(self_speed_info, all_objects_info)

                # print("outputs", outputs.shape)
                # print(outputs)
                # exit()
                # loss = criterion(outputs, action_label)
                # print(loss.item(), total_batch)
                moveable_object_info = all_objects_info[0][img_index]
                lane_info = all_objects_info[1][img_index]
                traffic_object_info = all_objects_info[2][img_index]

                one_img_predict_action = torch.sigmoid(outputs[img_index]) > 0.5
                one_img_moveable_object = moveable_object_alpha[img_index]
                one_img_lane_alpha = lane_alpha[img_index]
                one_img_traffic_light_alpha = traffic_light_alpha[img_index]
                one_img_action_label = action_label[img_index]
                # f1_side_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')

                # print(loss.item(), f1_side_action, total_batch)

    # print("moveable_object_alpha", moveable_object_alpha.shape)
    # print("lane_info", lane_info.shape)
    # print("traffic_object_info", traffic_object_info.shape)
    # exit()
    return  one_img_predict_action, one_img_action_label, one_img_moveable_object, one_img_lane_alpha, one_img_traffic_light_alpha, moveable_object_info, lane_info, traffic_object_info

def save_RNNattention(save_csv_folder_path, model_path, train_iter, config, device, max_moveable_object_maxnum,
                        max_lane_object_maxnum, max_traffic_light_maxnum):
    model = Model(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    f1_side_action_list = []
    with torch.no_grad():
        for i, (self_speed_info, all_objects_info, action_label, img_name_list) in enumerate(train_iter):
            # print("all_objects_info", all_objects_info[0].shape)
            # exit()
            outputs, moveable_object_alpha, lane_alpha, traffic_light_alpha = model.show_attention(self_speed_info, all_objects_info)
            for img_index, _ in enumerate(img_name_list):

                # print("all_object_alpha", all_object_alpha.shape)
                # print(outputs)
                # exit()
                # loss = criterion(outputs, action_label)
                # print(loss.item(), total_batch)
                moveable_object_info = all_objects_info[0][img_index]
                lane_info = all_objects_info[1][img_index]
                traffic_object_info = all_objects_info[2][img_index]

                one_img_predict_action = torch.sigmoid(outputs[img_index]) > 0.5


                one_img_moveable_object_alpha = moveable_object_alpha[img_index]
                one_img_lane_alpha = lane_alpha[img_index]
                one_img_traffic_light_alpha = traffic_light_alpha[img_index]

                one_img_action_label = action_label[img_index]

                save_csv_path = os.path.join(save_csv_folder_path, img_name_list[img_index])
                save_a_attention_for_a_img(save_csv_path, one_img_predict_action, one_img_action_label, one_img_moveable_object_alpha,
                 one_img_lane_alpha, one_img_traffic_light_alpha, moveable_object_info, lane_info, traffic_object_info)

                # exit()
                # f1_side_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')

                # print(loss.item(), f1_side_action, total_batch)