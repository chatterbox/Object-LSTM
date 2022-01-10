# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextRNN'
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
        self.num_epochs = 3000                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        # print(self.embed) # pre_trained : 300 # random: 300 当我们选择用词向量的时候，我们就不会再使用这里的字向量
        # exit()
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # self_speed_info
        self.self_speed_fc = nn.Linear(config.self_speed_info_size, config.self_speed_middle_info_size)

        # moveable_object_info_lstm
        self.moveable_object_lstm = nn.LSTM(config.object_info_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.moveable_object_fc = nn.Linear(config.hidden_size * 2, config.object_middle_info_size)

        # lane_info_lstm
        self.lane_info_lstm = nn.LSTM(config.lane_info_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lane_info_fc = nn.Linear(config.hidden_size * 2, config.lane_middle_info_size)

        # traffic_info_lstm
        self.traffic_info_lstm = nn.LSTM(config.traffic_info_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
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

    # def forward(self, self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img):
    #     self_speed_tensor_info_list = self_speed_info
    #     all_object_info_tuple = all_objects_info
    #     # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
    #     self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)
    #
    #     moveable_object_lstm_out, (moveable_object_lstm_hn, _) = self.moveable_object_lstm(all_object_info_tuple[0])
    #     # print(moveable_object_lstm_out[:, -1, :].shape)
    #     # exit()
    #     moveable_object_out_hn = torch.cat((moveable_object_lstm_hn[2], moveable_object_lstm_hn[3]), -1)
    #     object_middle_info = self.moveable_object_fc( moveable_object_out_hn )
    #
    #
    #     lane_lstm_out, (lane_lstm_hn, _) =  self.lane_info_lstm(all_object_info_tuple[1])
    #     lane_out_hn = torch.cat((lane_lstm_hn[2], lane_lstm_hn[3]), -1)
    #     lane_middle_info = self.lane_info_fc(lane_out_hn)
    #
    #
    #     traffic_light_lstm_out, (traffic_light_lstm_hn, _) = self.traffic_info_lstm(all_object_info_tuple[2])
    #     traffic_light_hn = torch.cat((traffic_light_lstm_hn[2], traffic_light_lstm_hn[3]), -1)
    #     traffic_light_middle_info = self.traffic_info_fc(traffic_light_hn)
    #
    #     over_all_middle_info = torch.cat( (self_speed_middle_info, object_middle_info, lane_middle_info, traffic_light_middle_info), dim=1)
    #     # print(self_speed_middle_info.shape, object_middle_info)
    #     # print(over_all_middle_info.shape)
    #     # exit()
    #     over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
    #     # print(over_all_middle_info.shape)
    #     middle_fc_feature =  self.drop(self.relu1(self.over_all_fc(over_all_middle_info)))
    #     output =   self.middle_fc1(middle_fc_feature)
    #     # output = self.middle_fc0(output)
    #     output =  self.middle_fc2(output)
    #
    #     # print(output.shape)
    #     # exit()
    #     return output

    def forward(self, self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img):

        moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img = three_kinds_of_object_num_in_an_img
        moveable_object_info, lane_object_info, traffic_light_object_info = all_objects_info
        # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img.shape)
        # print("moveable_object_info", moveable_object_info.shape)
        # exit()
        self_speed_tensor_info_list = self_speed_info
        all_object_info_tuple = all_objects_info
        # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]

        self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)

        def changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info, moveable_object_lstm, moveable_object_fc):
            _, moveable_object_num_sort = torch.sort(moveable_object_num_for_an_img, dim=0,
                                                 descending=True)  # 长度从长到短排序（index）
            _, moveable_object_num_unsort = torch.sort(moveable_object_num_sort)  # 排序后，原序列的 index

            moveable_object_out = torch.index_select(moveable_object_info, 0, moveable_object_num_sort)
            moveable_object_num_for_an_img = list(moveable_object_num_for_an_img[moveable_object_num_sort])
            # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img)
            moveable_object_out = nn.utils.rnn.pack_padded_sequence(moveable_object_out, moveable_object_num_for_an_img,
                                                                batch_first=True)
            _, (moveable_object_out_hn, _) = moveable_object_lstm(moveable_object_out)
            moveable_object_out = torch.cat((moveable_object_out_hn[2], moveable_object_out_hn[3]), -1)
            moveable_object_out = moveable_object_out.index_select(0, moveable_object_num_unsort)
            moveable_object_middle_info = moveable_object_fc(moveable_object_out)
            return moveable_object_middle_info

        moveable_object_middle_info = changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info, self.moveable_object_lstm, self.moveable_object_fc)
        lane_object_middle_info = changeable_input_lstm_out(self, lane_object_num_for_an_img, lane_object_info, self.lane_info_lstm, self.lane_info_fc)
        traffic_light_middle_info = changeable_input_lstm_out(self, traffic_light_object_num_for_an_img, traffic_light_object_info, self.traffic_info_lstm, self.traffic_info_fc)

        over_all_middle_info = torch.cat( (self_speed_middle_info, moveable_object_middle_info, lane_object_middle_info, traffic_light_middle_info), dim=1)
        # print(self_speed_middle_info.shape, object_middle_info)
        # print(over_all_middle_info.shape)
        # exit()
        over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
        # print(over_all_middle_info.shape)
        middle_fc_feature =  self.drop(self.relu1(self.over_all_fc(over_all_middle_info)))
        output =   self.middle_fc1(middle_fc_feature)
        # output = self.middle_fc0(output)
        output =  self.middle_fc2(output)

        return output
