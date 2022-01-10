
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
        # self.model_name = 'TextRNN_Att_3RNN_1attention_changeable_input(new_lane_plus_sp)'
        self.model_name = 'TextRNN_Att_3RNN_1attention_changeable_input(RA_test_selfspd)'
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
        self.lane_info_size = 5
        self.traffic_info_size = 5

        self.all_object_middle_info_size = 150
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
    def __init__(self, config, self_speed_flag):
        super(Model, self).__init__()

        self.all_object_attention_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))

        self.moveable_object_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.lane_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.traffic_light_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))

        self.all_object_fc = nn.Linear(config.hidden_size * 2, config.all_object_middle_info_size)


        self.self_speed_fc = nn.Linear(config.self_speed_info_size, config.self_speed_middle_info_size)

        # moveable_object_info_lstm
        self.moveable_object_lstm = nn.LSTM(config.object_info_size, config.hidden_size, config.num_layers,
                                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh_moveable_object = nn.Tanh()
        # self.moveable_object_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))


        # lane_info_lstm
        self.lane_info_lstm = nn.LSTM(config.lane_info_size, config.hidden_size, config.num_layers,
                                      bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh_lane = nn.Tanh()
        # self.lane_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))


        # traffic_info_lstm
        self.traffic_info_lstm = nn.LSTM(config.traffic_info_size, config.hidden_size, config.num_layers,
                                         bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh_traffic_light = nn.Tanh()
        # self.traffic_light_weight = nn.Parameter(torch.zeros(config.hidden_size * 2))


        # over_all_info_fc
        self.middle_fc_feature1 = 100
        self.middle_fc_feature2 = 50
        self.middle_fc_feature0 = 10

        self.drop = nn.Dropout(p=0.5)
        self.relu1 = nn.ReLU(inplace=True)
        over_all_input_size = 256
        self.middle_fc_feature1_out = 50
        if self_speed_flag == True:
            self.middle_fc_feature2 = 50 + 10

        else:
            self.middle_fc_feature2 = 50


        self.over_all_fc = nn.Linear(over_all_input_size, self.middle_fc_feature1)
        self.middle_fc1 = nn.Linear(self.middle_fc_feature1, self.middle_fc_feature1_out)
        self.middle_fc2 = nn.Linear(self.middle_fc_feature2, config.num_classes)


    # def forward(self, self_speed_info, all_objects_info):
    #
    #     self_speed_tensor_info_list = self_speed_info
    #     all_object_info_tuple = all_objects_info
    #     # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
    #     self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)
    #
    #     moveable_object_lstm_out, _ = self.moveable_object_lstm(all_object_info_tuple[0])
    #     moveable_object_M = self.tanh_moveable_object(moveable_object_lstm_out)  # [128, 32, 256]
    #
    #
    #     lane_lstm_out, _ = self.lane_info_lstm(all_object_info_tuple[1])
    #     lane_M = self.tanh_lane(lane_lstm_out)  # [128, 32, 256]
    #
    #
    #     traffic_light_lstm_out, _ = self.traffic_info_lstm(all_object_info_tuple[2])
    #     traffic_light_M = self.tanh_traffic_light(traffic_light_lstm_out)  # [128, 32, 256]
    #
    #
    #     all_object_M = torch.cat((moveable_object_M, lane_M, traffic_light_M), dim=1)
    #     all_object_alpha = F.softmax(torch.matmul(all_object_M, self.all_object_attention_weight), dim=1).unsqueeze(
    #         -1)  # [128, 32, 1]
    #
    #     all_object_lstm_out = torch.cat((moveable_object_lstm_out, lane_lstm_out, traffic_light_lstm_out), dim=1)
    #     all_object_out = all_object_lstm_out * all_object_alpha  # [128, 32, 256]
    #     all_object_out = torch.sum(all_object_out, 1)  # [50, 256]
    #     all_object_out = F.relu(all_object_out)
    #     object_middle_info = self.all_object_fc(all_object_out)
    #     # print("all_object_out.shape", all_object_out.shape)
    #     # exit()
    #
    #     over_all_middle_info = torch.cat(
    #         (self_speed_middle_info, object_middle_info), dim=1)
    #     # print(self_speed_middle_info.shape, object_middle_info)
    #     # print(over_all_middle_info.shape)
    #     # exit()
    #     over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
    #     # print(over_all_middle_info.shape)
    #     middle_fc_feature = self.drop(self.relu1(self.over_all_fc(over_all_middle_info)))
    #     output = self.middle_fc1(middle_fc_feature)
    #     # output = self.middle_fc0(output)
    #     output = self.middle_fc2(output)
    #
    #     return output

    # def forward(self, self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img):
    #
    #     moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img = three_kinds_of_object_num_in_an_img
    #     moveable_object_info, lane_object_info, traffic_light_object_info = all_objects_info
    #     # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img.sum())
    #     # print("moveable_object_info", moveable_object_info.shape)
    #     # exit()
    #     self_speed_tensor_info_list = self_speed_info
    #     all_object_info_tuple = all_objects_info
    #     # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
    #
    #     self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)
    #
    #     def changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info, moveable_object_lstm):
    #         _, moveable_object_num_sort = torch.sort(moveable_object_num_for_an_img, dim=0,
    #                                              descending=True)  # 长度从长到短排序（index）
    #         _, moveable_object_num_unsort = torch.sort(moveable_object_num_sort)  # 排序后，原序列的 index
    #         moveable_object_out = torch.index_select(moveable_object_info, 0, moveable_object_num_sort)
    #         moveable_object_num_for_an_img = list(moveable_object_num_for_an_img[moveable_object_num_sort])
    #         # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img)
    #         moveable_object_out = nn.utils.rnn.pack_padded_sequence(moveable_object_out, moveable_object_num_for_an_img,
    #                                                             batch_first=True)
    #         moveable_object_out, (moveable_object_out_hn, _) = moveable_object_lstm(moveable_object_out)
    #
    #         moveable_object_out, out_len = nn.utils.rnn.pad_packed_sequence(moveable_object_out, batch_first=True)
    #         # print(moveable_object_num_for_an_img)
    #         # print(moveable_object_out.shape)
    #         # print(moveable_object_out[1][22]) # 有内容
    #         # print(moveable_object_out[1][23]) # 全为0
    #         # exit()
    #         moveable_object_out = moveable_object_out.index_select(0, moveable_object_num_unsort)
    #         return moveable_object_out
    #
    #     moveable_object_out = changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info, self.moveable_object_lstm, )
    #     lane_object_out = changeable_input_lstm_out(self, lane_object_num_for_an_img, lane_object_info, self.lane_info_lstm)
    #     traffic_light_out = changeable_input_lstm_out(self, traffic_light_object_num_for_an_img, traffic_light_object_info, self.traffic_info_lstm)
    #
    #
    #     moveable_object_importance = torch.matmul(moveable_object_out, self.moveable_object_weight)
    #
    #     lane_object_importance = torch.matmul(lane_object_out, self.lane_weight)
    #     traffic_object_importance = torch.matmul(traffic_light_out, self.traffic_light_weight)
    #
    #
    #     all_object_importance = torch.cat((moveable_object_importance, lane_object_importance, traffic_object_importance), dim=1) # [50,x]
    #
    #     all_object_out = torch.cat((moveable_object_out, lane_object_out, traffic_light_out), dim=1) # [50,x]
    #
    #     alpha = F.softmax(all_object_importance, dim=1).unsqueeze(-1)  # [50, x, 1]
    #
    #     out = all_object_out * alpha  # [128, 32, 256]
    #     out = torch.sum(out, 1)  # [128, 256]
    #
    #
    #     # print(self_speed_middle_info.shape, object_middle_info)
    #     # print(over_all_middle_info.shape)
    #     # exit()
    #     # over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
    #     # print(over_all_middle_info.shape)
    #     middle_fc_feature =  self.drop(self.relu1(self.over_all_fc(out)))
    #     output =   self.middle_fc1(middle_fc_feature)
    #     # output = self.middle_fc0(output)
    #     output =  self.middle_fc2(output)
    #
    #
    #     return output

    def forward(self, self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img, self_speed_flag):

            moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img = three_kinds_of_object_num_in_an_img
            moveable_object_info, lane_object_info, traffic_light_object_info = all_objects_info
            # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img.sum())
            # print("moveable_object_info", moveable_object_info.shape)
            # exit()
            self_speed_tensor_info_list = self_speed_info
            all_object_info_tuple = all_objects_info
            # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]

            self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)

            def changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info,
                                          moveable_object_lstm):
                _, moveable_object_num_sort = torch.sort(moveable_object_num_for_an_img, dim=0,
                                                         descending=True)  # 长度从长到短排序（index）
                _, moveable_object_num_unsort = torch.sort(moveable_object_num_sort)  # 排序后，原序列的 index
                moveable_object_out = torch.index_select(moveable_object_info, 0, moveable_object_num_sort)
                moveable_object_num_for_an_img = list(moveable_object_num_for_an_img[moveable_object_num_sort])
                # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img)
                moveable_object_out = nn.utils.rnn.pack_padded_sequence(moveable_object_out,
                                                                        moveable_object_num_for_an_img,
                                                                        batch_first=True)
                moveable_object_out, (moveable_object_out_hn, _) = moveable_object_lstm(moveable_object_out)

                moveable_object_out, out_len = nn.utils.rnn.pad_packed_sequence(moveable_object_out, batch_first=True)
                moveable_object_out = moveable_object_out.index_select(0, moveable_object_num_unsort)
                return moveable_object_out

            moveable_object_out = changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info,
                                                            self.moveable_object_lstm, )
            lane_object_out = changeable_input_lstm_out(self, lane_object_num_for_an_img, lane_object_info,
                                                        self.lane_info_lstm)
            traffic_light_out = changeable_input_lstm_out(self, traffic_light_object_num_for_an_img,
                                                          traffic_light_object_info, self.traffic_info_lstm)

            def each_kind_object_in_a_img(in_batch_flag, moveable_object_num_for_an_img, moveable_object_out):
                a_img_moveable_object = moveable_object_out[in_batch_flag,
                                        :int(moveable_object_num_for_an_img[in_batch_flag]), :]
                a_img_moveable_object_importance = torch.matmul(a_img_moveable_object, self.moveable_object_weight)

                a_img_moveable_object_alpha = F.softmax(a_img_moveable_object_importance).unsqueeze(
                    -1)  # [50, x, 1]
                # a_img_moveable_object_attentioned = a_img_moveable_object * a_img_moveable_object_alpha
                return a_img_moveable_object, a_img_moveable_object_importance

            def attentioned_for_a_img(moveable_object_num_for_an_img, moveable_object_out, lane_object_num_for_an_img,
                                      lane_object_out, traffic_light_object_num_for_an_img, traffic_light_out):
                all_object_alpha_in_a_batch = []
                a_img_moveable_object, a_img_moveable_object_importance = each_kind_object_in_a_img(0,
                                                                                                    moveable_object_num_for_an_img,
                                                                                                    moveable_object_out)
                a_img_lane_object, a_img_lane_object_importance = each_kind_object_in_a_img(0,
                                                                                            lane_object_num_for_an_img,
                                                                                            lane_object_out)
                a_img_traffic_light_object, a_img_traffic_light_object_importance = each_kind_object_in_a_img(0,
                                                                                                              traffic_light_object_num_for_an_img,
                                                                                                              traffic_light_out)

                all_object_importance = torch.cat((a_img_moveable_object_importance, a_img_lane_object_importance,
                                                   a_img_traffic_light_object_importance), dim=0)

                all_object_out = torch.cat(
                    (a_img_moveable_object, a_img_lane_object, a_img_traffic_light_object),
                    dim=0)
                a_img_all_object_alpha = F.softmax(all_object_importance).unsqueeze(-1)  # [50, x, 1]

                # a_img_all_object_alpha = all_object_importance.unsqueeze(-1)  # [50, x, 1]
                # a_img_all_object_alpha = F.relu(a_img_all_object_alpha)
                a_img_all_object_attentioned = all_object_out * a_img_all_object_alpha
                # print(a_img_moveable_object_alpha.shape)
                # print(a_img_lane_object_alpha.shape)
                # print(a_img_traffic_light_object_alpha.shape)
                # exit()
                all_object_alpha_in_a_batch.append(a_img_all_object_alpha)

                out_in_a_batch = torch.sum(a_img_all_object_attentioned, 0)  # [128, 256]
                # print(out_in_a_batch.shape)
                out_in_a_batch = out_in_a_batch.unsqueeze(0)

                for in_batch_flag in range(1, moveable_object_num_for_an_img.shape[0]):
                    a_img_moveable_object, a_img_moveable_object_importance = each_kind_object_in_a_img(in_batch_flag,
                                                                                                        moveable_object_num_for_an_img,
                                                                                                        moveable_object_out)
                    a_img_lane_object, a_img_lane_object_importance = each_kind_object_in_a_img(in_batch_flag,
                                                                                                lane_object_num_for_an_img,
                                                                                                lane_object_out)
                    a_img_traffic_light_object, a_img_traffic_light_object_importance = each_kind_object_in_a_img(
                        in_batch_flag, traffic_light_object_num_for_an_img, traffic_light_out)

                    # print(moveable_object_out_in_a_img.shape)
                    # print(lane_object_out_in_a_img.shape)
                    # print(traffic_light_object_out_in_a_img.shape)
                    all_object_importance = torch.cat((a_img_moveable_object_importance, a_img_lane_object_importance,
                                                       a_img_traffic_light_object_importance), dim=0)

                    all_object_out = torch.cat(
                        (a_img_moveable_object, a_img_lane_object, a_img_traffic_light_object),
                        dim=0)
                    a_img_all_object_alpha = F.softmax(all_object_importance).unsqueeze(-1)  # [50, x, 1]
                    # a_img_all_object_alpha = all_object_importance.unsqueeze(-1)  # [50, x, 1]
                    # a_img_all_object_alpha = F.relu(a_img_all_object_alpha)
                    a_img_all_object_attentioned = all_object_out * a_img_all_object_alpha
                    # print(a_img_moveable_object_alpha.shape)
                    # print(a_img_lane_object_alpha.shape)
                    # print(a_img_traffic_light_object_alpha.shape)
                    # exit()
                    # all_object_alpha_in_a_batch.append(a_img_all_object_alpha)

                    out_in_a_img = torch.sum(a_img_all_object_attentioned, 0)  # [128, 256]
                    # print(out_in_a_batch.shape)
                    out_in_a_img = out_in_a_img.unsqueeze(0)

                    out_in_a_batch = torch.cat((out_in_a_batch, out_in_a_img), dim=0)

                    all_object_alpha_in_a_batch.append(a_img_all_object_alpha)
                # print(out_in_a_batch.shape)
                return out_in_a_batch, all_object_alpha_in_a_batch

            # print(moveable_object_num_for_an_img)
            # print(moveable_object_out.shape)  # 有内容
            # print(moveable_object_out[0][16]) # 有内容
            # print(moveable_object_out[0][17]) # 全为0
            out_in_a_batch, all_object_alpha_in_a_batch = attentioned_for_a_img(moveable_object_num_for_an_img,
                                                                                moveable_object_out,
                                                                                lane_object_num_for_an_img,
                                                                                lane_object_out,
                                                                                traffic_light_object_num_for_an_img,
                                                                                traffic_light_out)



            middle_fc_feature = self.drop(self.relu1(self.over_all_fc(out_in_a_batch)))
            output = self.middle_fc1(middle_fc_feature)
            # output = self.middle_fc0(output)
            # print(self_speed_middle_info.shape, output.shape)
            # exit()
            if self_speed_flag == True:
                output = torch.cat((self_speed_middle_info, output), 1)
            else:
                pass
            output = self.middle_fc2(output)

            # moveable_object_importance = torch.matmul(moveable_object_out, self.moveable_object_weight)
            # lane_object_importance = torch.matmul(lane_object_out, self.lane_weight)
            # traffic_object_importance = torch.matmul(traffic_light_out, self.traffic_light_weight)
            #
            # all_object_importance = torch.cat(
            #     (moveable_object_importance, lane_object_importance, traffic_object_importance), dim=1)  # [50,x]
            #
            # all_object_out = torch.cat((moveable_object_out, lane_object_out, traffic_light_out), dim=1)  # [50,x]
            #
            # alpha = F.softmax(all_object_importance, dim=1).unsqueeze(-1)  # [50, x, 1]
            # print(all_object_importance[0][:29])
            # print(alpha.shape)
            # print(alpha[0][:29])
            # exit()
            # out = all_object_out * alpha  # [128, 32, 256]
            # out = torch.sum(out, 1)  # [128, 256]

            # print(self_speed_middle_info.shape, object_middle_info)
            # print(over_all_middle_info.shape)
            # exit()
            # over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
            # print(over_all_middle_info.shape)

            # print(alpha.shape)
            # print(moveable_object_out.shape)
            # print(lane_object_out.shape)
            # print(traffic_light_out.shape)
            # print("*"*100)
            # print("*" * 100)

            return output, all_object_alpha_in_a_batch, three_kinds_of_object_num_in_an_img

    def show_attention(self, self_speed_info, all_objects_info):

        self_speed_tensor_info_list = self_speed_info
        all_object_info_tuple = all_objects_info
        # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)

        moveable_object_lstm_out, _ = self.moveable_object_lstm(all_object_info_tuple[0])
        moveable_object_M = self.tanh_moveable_object(moveable_object_lstm_out)  # [128, 32, 256]


        lane_lstm_out, _ = self.lane_info_lstm(all_object_info_tuple[1])
        lane_M = self.tanh_lane(lane_lstm_out)  # [128, 32, 256]


        traffic_light_lstm_out, _ = self.traffic_info_lstm(all_object_info_tuple[2])
        traffic_light_M = self.tanh_traffic_light(traffic_light_lstm_out)  # [128, 32, 256]


        all_object_M = torch.cat((moveable_object_M, lane_M, traffic_light_M), dim=1)
        all_object_alpha = F.softmax(torch.matmul(all_object_M, self.all_object_attention_weight), dim=1).unsqueeze(
            -1)  # [128, 32, 1]

        all_object_lstm_out = torch.cat((moveable_object_lstm_out, lane_lstm_out, traffic_light_lstm_out), dim=1)
        all_object_out = all_object_lstm_out * all_object_alpha  # [128, 32, 256]
        all_object_out = torch.sum(all_object_out, 1)  # [50, 256]
        object_middle_info = self.all_object_fc(all_object_out)
        # print("all_object_out.shape", all_object_out.shape)
        # exit()

        over_all_middle_info = torch.cat(
            (self_speed_middle_info, object_middle_info), dim=1)
        # print(self_speed_middle_info.shape, object_middle_info)
        # print(over_all_middle_info.shape)
        # exit()
        over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
        # print(over_all_middle_info.shape)
        middle_fc_feature = self.drop(self.relu1(self.over_all_fc(over_all_middle_info)))
        output = self.middle_fc1(middle_fc_feature)
        # output = self.middle_fc0(output)
        output = self.middle_fc2(output)

        return output, all_object_alpha

    def show_attention_changeable_input(self, self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img):
        moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img = three_kinds_of_object_num_in_an_img
        moveable_object_info, lane_object_info, traffic_light_object_info = all_objects_info
        # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img.sum())
        # print("moveable_object_info", moveable_object_info.shape)
        # exit()
        self_speed_tensor_info_list = self_speed_info
        all_object_info_tuple = all_objects_info
        # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]

        self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)

        def changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info, moveable_object_lstm):
            _, moveable_object_num_sort = torch.sort(moveable_object_num_for_an_img, dim=0,
                                                     descending=True)  # 长度从长到短排序（index）
            _, moveable_object_num_unsort = torch.sort(moveable_object_num_sort)  # 排序后，原序列的 index
            moveable_object_out = torch.index_select(moveable_object_info, 0, moveable_object_num_sort)
            moveable_object_num_for_an_img = list(moveable_object_num_for_an_img[moveable_object_num_sort])
            # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img)
            moveable_object_out = nn.utils.rnn.pack_padded_sequence(moveable_object_out, moveable_object_num_for_an_img,
                                                                    batch_first=True)
            moveable_object_out, (moveable_object_out_hn, _) = moveable_object_lstm(moveable_object_out)

            moveable_object_out, out_len = nn.utils.rnn.pad_packed_sequence(moveable_object_out, batch_first=True)
            moveable_object_out = moveable_object_out.index_select(0, moveable_object_num_unsort)
            return moveable_object_out

        moveable_object_out = changeable_input_lstm_out(self, moveable_object_num_for_an_img, moveable_object_info,
                                                        self.moveable_object_lstm, )
        lane_object_out = changeable_input_lstm_out(self, lane_object_num_for_an_img, lane_object_info,
                                                    self.lane_info_lstm)
        traffic_light_out = changeable_input_lstm_out(self, traffic_light_object_num_for_an_img,
                                                      traffic_light_object_info, self.traffic_info_lstm)

        def each_kind_object_in_a_img(in_batch_flag, moveable_object_num_for_an_img, moveable_object_out):
            a_img_moveable_object = moveable_object_out[in_batch_flag,
                                    :int(moveable_object_num_for_an_img[in_batch_flag]), :]
            a_img_moveable_object_importance = torch.matmul(a_img_moveable_object, self.moveable_object_weight)

            a_img_moveable_object_alpha = F.softmax(a_img_moveable_object_importance).unsqueeze(
                -1)  # [50, x, 1]
            a_img_moveable_object_attentioned = a_img_moveable_object * a_img_moveable_object_alpha
            return a_img_moveable_object, a_img_moveable_object_importance

        def attentioned_for_a_img(moveable_object_num_for_an_img, moveable_object_out, lane_object_num_for_an_img, lane_object_out, traffic_light_object_num_for_an_img, traffic_light_out):
            all_object_alpha_in_a_batch = []
            a_img_moveable_object, a_img_moveable_object_importance = each_kind_object_in_a_img(0, moveable_object_num_for_an_img,
                                                                     moveable_object_out)
            a_img_lane_object, a_img_lane_object_importance = each_kind_object_in_a_img(0, lane_object_num_for_an_img,
                                                                 lane_object_out)
            a_img_traffic_light_object, a_img_traffic_light_object_importance = each_kind_object_in_a_img(0,
                                                                          traffic_light_object_num_for_an_img,
                                                                          traffic_light_out)

            all_object_importance = torch.cat((a_img_moveable_object_importance, a_img_lane_object_importance, a_img_traffic_light_object_importance), dim=0)

            all_object_out = torch.cat(
                (a_img_moveable_object, a_img_lane_object, a_img_traffic_light_object),
                dim=0)
            a_img_all_object_alpha = F.softmax(all_object_importance).unsqueeze(-1)  # [50, x, 1]
            # a_img_all_object_alpha = all_object_importance.unsqueeze(-1)  # [50, x, 1]
            # a_img_all_object_alpha = F.relu(a_img_all_object_alpha)
            a_img_all_object_attentioned = all_object_out * a_img_all_object_alpha
            # print(a_img_moveable_object_alpha.shape)
            # print(a_img_lane_object_alpha.shape)
            # print(a_img_traffic_light_object_alpha.shape)
            # exit()
            all_object_alpha_in_a_batch.append(a_img_all_object_alpha)

            out_in_a_batch = torch.sum(a_img_all_object_attentioned, 0)  # [128, 256]
            # print(out_in_a_batch.shape)
            out_in_a_batch = out_in_a_batch.unsqueeze(0)

            for in_batch_flag in range(1, moveable_object_num_for_an_img.shape[0]):
                a_img_moveable_object, a_img_moveable_object_importance = each_kind_object_in_a_img(in_batch_flag, moveable_object_num_for_an_img, moveable_object_out)
                a_img_lane_object, a_img_lane_object_importance = each_kind_object_in_a_img(in_batch_flag, lane_object_num_for_an_img, lane_object_out)
                a_img_traffic_light_object, a_img_traffic_light_object_importance = each_kind_object_in_a_img(in_batch_flag, traffic_light_object_num_for_an_img, traffic_light_out)

                # print(moveable_object_out_in_a_img.shape)
                # print(lane_object_out_in_a_img.shape)
                # print(traffic_light_object_out_in_a_img.shape)
                all_object_importance = torch.cat((a_img_moveable_object_importance, a_img_lane_object_importance,
                                                   a_img_traffic_light_object_importance), dim=0)

                all_object_out = torch.cat(
                    (a_img_moveable_object, a_img_lane_object, a_img_traffic_light_object),
                    dim=0)
                a_img_all_object_alpha = F.softmax(all_object_importance).unsqueeze(-1)  # [50, x, 1]
                # a_img_all_object_alpha = F.relu(a_img_all_object_alpha)
                a_img_all_object_attentioned = all_object_out * a_img_all_object_alpha
                # print(a_img_moveable_object_alpha.shape)
                # print(a_img_lane_object_alpha.shape)
                # print(a_img_traffic_light_object_alpha.shape)
                # exit()
                # all_object_alpha_in_a_batch.append(a_img_all_object_alpha)

                out_in_a_img = torch.sum(a_img_all_object_attentioned, 0)  # [128, 256]
                # print(out_in_a_batch.shape)
                out_in_a_img = out_in_a_img.unsqueeze(0)

                out_in_a_batch = torch.cat((out_in_a_batch, out_in_a_img), dim=0)

                all_object_alpha_in_a_batch.append(a_img_all_object_alpha)
            # print(out_in_a_batch.shape)
            return out_in_a_batch, all_object_alpha_in_a_batch

        # print(moveable_object_num_for_an_img)
        # print(moveable_object_out.shape)  # 有内容
        # print(moveable_object_out[0][16]) # 有内容
        # print(moveable_object_out[0][17]) # 全为0
        out_in_a_batch, all_object_alpha_in_a_batch = attentioned_for_a_img(moveable_object_num_for_an_img, moveable_object_out, lane_object_num_for_an_img, lane_object_out, traffic_light_object_num_for_an_img, traffic_light_out)

        # object_info_plus_self_speed = torch.cat((self_speed_middle_info, out_in_a_batch), 1)
        object_info_plus_self_speed = out_in_a_batch
        middle_fc_feature = self.drop(self.relu1(self.over_all_fc(object_info_plus_self_speed)))

        # middle_fc_feature = self.drop(self.relu1(self.over_all_fc(middle_fc_feature)))
        output = self.middle_fc1(middle_fc_feature)
        # output = self.middle_fc0(output)
        output = self.middle_fc2(output)

        # moveable_object_importance = torch.matmul(moveable_object_out, self.moveable_object_weight)
        # lane_object_importance = torch.matmul(lane_object_out, self.lane_weight)
        # traffic_object_importance = torch.matmul(traffic_light_out, self.traffic_light_weight)
        #
        # all_object_importance = torch.cat(
        #     (moveable_object_importance, lane_object_importance, traffic_object_importance), dim=1)  # [50,x]
        #
        # all_object_out = torch.cat((moveable_object_out, lane_object_out, traffic_light_out), dim=1)  # [50,x]
        #
        # alpha = F.softmax(all_object_importance, dim=1).unsqueeze(-1)  # [50, x, 1]
        # print(all_object_importance[0][:29])
        # print(alpha.shape)
        # print(alpha[0][:29])
        # exit()
        # out = all_object_out * alpha  # [128, 32, 256]
        # out = torch.sum(out, 1)  # [128, 256]

        # print(self_speed_middle_info.shape, object_middle_info)
        # print(over_all_middle_info.shape)
        # exit()
        # over_all_middle_info = over_all_middle_info.view(over_all_middle_info.size(0), -1)
        # print(over_all_middle_info.shape)


        # print(alpha.shape)
        # print(moveable_object_out.shape)
        # print(lane_object_out.shape)
        # print(traffic_light_out.shape)
        # print("*"*100)
        # print("*" * 100)

        return output, all_object_alpha_in_a_batch, three_kinds_of_object_num_in_an_img

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
                outputs, all_object_alpha = model.show_attention(self_speed_info, all_objects_info)

                # print("all_object_alpha", all_object_alpha.shape)
                # print(outputs)
                # exit()
                # loss = criterion(outputs, action_label)
                # print(loss.item(), total_batch)
                moveable_object_info = all_objects_info[0][img_index]
                lane_info = all_objects_info[1][img_index]
                traffic_object_info = all_objects_info[2][img_index]

                one_img_predict_action = torch.sigmoid(outputs[img_index]) > 0.5

                one_img_moveable_object = all_object_alpha[img_index][:max_moveable_object_maxnum]
                one_img_lane_alpha = all_object_alpha[img_index][max_moveable_object_maxnum:max_moveable_object_maxnum + max_lane_object_maxnum]
                one_img_traffic_light_alpha = all_object_alpha[img_index][max_moveable_object_maxnum + max_lane_object_maxnum:]

                one_img_action_label = action_label[img_index]
                break
                # f1_side_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')

                # print(loss.item(), f1_side_action, total_batch)

    # print("moveable_object_alpha", one_img_moveable_object.shape)
    # print("lane_info", one_img_lane_alpha.shape)
    # print("traffic_object_info", one_img_traffic_light_alpha.shape)
    # exit()
    return  one_img_predict_action, one_img_action_label, one_img_moveable_object, one_img_lane_alpha, one_img_traffic_light_alpha, moveable_object_info, lane_info, traffic_object_info


def save_a_attention_for_a_img(save_csv_path, one_img_predict_action, one_img_action_label, three_kinds_of_objects_num, one_img_moveable_object_alpha, one_img_lane_alpha, one_img_traffic_light_alpha, moveable_object_info, lane_info, traffic_object_info):
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
        writer.writerow(three_kinds_of_objects_num)

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

def save_RNNattention(save_csv_folder_path, model_path, train_iter, config, device, self_speed_flag):
    model = Model(config, self_speed_flag).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    f1_side_action_list = []
    with torch.no_grad():
        for i, (self_speed_info, all_objects_info, action_label, img_name_list, three_kinds_of_object_num_in_an_img) in enumerate(train_iter):

            outputs, all_object_alpha, three_kinds_of_object_num_in_an_img = model(self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img, self_speed_flag)
            # print(len(all_object_alpha))
            # print(all_object_alpha[0].shape)
            # print(three_kinds_of_object_num_in_an_img[0][0])
            # print(three_kinds_of_object_num_in_an_img[1][0])
            # print(three_kinds_of_object_num_in_an_img[2][0])
            # print(all_object_alpha[1].shape)
            # print(three_kinds_of_object_num_in_an_img[0][1])
            # print(three_kinds_of_object_num_in_an_img[1][1])
            # print(three_kinds_of_object_num_in_an_img[2][1])
            # exit()
            for img_index, _ in enumerate(img_name_list):
                moveable_object_num = int(three_kinds_of_object_num_in_an_img[0][img_index])
                lane_object_num = int(three_kinds_of_object_num_in_an_img[1][img_index])
                traffic_light_object_num = int(three_kinds_of_object_num_in_an_img[2][img_index])
                # print("all_object_alpha", all_object_alpha.shape)
                # print(outputs)
                # exit()
                # loss = criterion(outputs, action_label)
                # print(loss.item(), total_batch)


                moveable_object_info = all_objects_info[0][img_index][:moveable_object_num]
                lane_info = all_objects_info[1][img_index][:lane_object_num]
                traffic_object_info = all_objects_info[2][img_index][:traffic_light_object_num]

                one_img_predict_action = torch.sigmoid(outputs[img_index]) > 0.5

                one_img_moveable_object_alpha = all_object_alpha[img_index][:moveable_object_num]
                one_img_lane_alpha = all_object_alpha[img_index][
                                     moveable_object_num:moveable_object_num + lane_object_num]
                one_img_traffic_light_alpha = all_object_alpha[img_index][
                                              moveable_object_num + lane_object_num:moveable_object_num + lane_object_num+traffic_light_object_num]




                # print(all_objects_info[0].shape)

                #
                # exit()
                one_img_action_label = action_label[img_index]
                save_csv_path = os.path.join(save_csv_folder_path, img_name_list[img_index])

                print("*"*100)
                print(all_object_alpha[img_index].shape)
                print(three_kinds_of_object_num_in_an_img[0][img_index])
                print(three_kinds_of_object_num_in_an_img[1][img_index])
                print(three_kinds_of_object_num_in_an_img[2][img_index])
                print(one_img_moveable_object_alpha.shape)
                print(moveable_object_info.shape)

                print(one_img_lane_alpha.shape)
                print(lane_info.shape)

                print(one_img_traffic_light_alpha.shape)
                print(traffic_object_info.shape)

                if one_img_moveable_object_alpha.shape[0] != moveable_object_info.shape[0] or one_img_lane_alpha.shape[0] != lane_info.shape[0] or one_img_traffic_light_alpha.shape[0] != traffic_object_info.shape[0]:
                    exit()
                # print(one_img_traffic_light_alpha)
                # exit()
                three_kinds_of_objects_num = (moveable_object_num, lane_object_num, traffic_light_object_num)
                save_a_attention_for_a_img(save_csv_path, one_img_predict_action, one_img_action_label, three_kinds_of_objects_num, one_img_moveable_object_alpha,
                 one_img_lane_alpha, one_img_traffic_light_alpha, moveable_object_info, lane_info, traffic_object_info)

                # exit()
                # f1_side_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')

                # print(loss.item(), f1_side_action, total_batch)

