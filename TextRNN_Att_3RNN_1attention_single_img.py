# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network

import os
import csv
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextRNN_Att_3RNN_1attention_changeable_input(new_lane)'
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

def find_padding_size_for_each_info(folder_path, object_settings_path):
    object_dict = {}
    with open(object_settings_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # intense_loss_line.append(float(line[0]))
            object_dict[line[0]] = line[1]
        next(reader, None)  # jump to the next line

    file_name_list = os.listdir(folder_path)
    moveable_object_maxnum = 0
    lane_object_maxnum = 0
    traffic_light_maxnum = 0
    for single_csv_file in file_name_list:
        single_csv_path = os.path.join(folder_path, single_csv_file)

        moveable_object_num = 0
        lane_object_num = 0
        traffic_light_num = 0
        # print("single_csv_path", single_csv_path)
        with open(single_csv_path) as csvfile:
            reader = csv.reader(csvfile)
            line_flag = 0
            for line in reader:
                if line_flag == 0:
                    self_speed_info = line
                    line_flag = line_flag + 1
                    continue

                object_name = line[0]
                object_serial_num = int(object_dict[object_name])

                if object_serial_num <= 7:
                    moveable_object_num = moveable_object_num + 1

                if object_serial_num > 7 and object_serial_num <= 11:
                    grid_string = line[1]
                    grid_string = grid_string[1:]
                    grid_string = grid_string[:-1]
                    grid_list = grid_string.split(",")

                    for each_grid in grid_list:
                        lane_object_num = lane_object_num + 1

                if object_serial_num > 11 and object_serial_num <= 16:
                    traffic_light_num = traffic_light_num + 1

        if moveable_object_num > moveable_object_maxnum:
            moveable_object_maxnum = moveable_object_num

        if lane_object_num > lane_object_maxnum:
            lane_object_maxnum = lane_object_num

        if traffic_light_num > traffic_light_maxnum:
            traffic_light_maxnum = traffic_light_num

    return moveable_object_maxnum, lane_object_maxnum, traffic_light_maxnum


def read_csv_file(single_csv_path):
    samples = []
    with open(single_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # intense_loss_line.append(float(line[0]))
            samples.append(line)
        next(reader, None)  # jump to the next line
    return samples


def float_list__to_int_list(float_list):
    new_list = []
    for i in float_list:
        new_list.append(int(i))
    return new_list

def one_img_csv_process(single_csv_path, single_csv_file, object_settings_path, action_label_csv_path, one_possible_list):

    object_dict = {}
    with open(object_settings_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # intense_loss_line.append(float(line[0]))
            object_dict[line[0]] = line[1]
        next(reader, None)  # jump to the next line
    single_imgs_contents = []

    moveable_object_list = []
    lane_object_list = []
    traffic_light_list = []

    # print("single_csv_path", single_csv_path)
    with open(single_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        line_flag = 0
        for line in reader:
            if line_flag == 0:
                self_speed_info_x = int(line[0])
                self_speed_info_y = int(line[1])
                self_speed_info = [self_speed_info_x, self_speed_info_y]
                # print(self_speed_info)
                # exit()
                line_flag = line_flag + 1
                continue

            object_name = line[0]
            object_serial_num = int(object_dict[object_name])
            # print(object_name)
            # exit()
            if object_serial_num <= 7:
                object_info = [int(object_dict[object_name])] + float_list__to_int_list(line[1:])
                # object_info = [int(object_dict[object_name]), float_list__to_int_list(line[1:])]
                moveable_object_list.append(object_info)
                # print("object_info", object_info)

            if object_serial_num > 7 and object_serial_num <= 11:
                start_end_point_string = line[1]
                # start_end_point_string = "[[241, 546, 579, 524], [579, 524, 867, 521], [867, 521, 1172, 536]]"
                start_end_point_string = start_end_point_string[1:]
                start_end_point_string = start_end_point_string[:-1]
                # print(start_end_point_string)
                start_end_point_list = start_end_point_string.split("], ")
                # start_end_point_list = start_end_point_list[:-1]
                # print(start_end_point_list)
                for flag, each_start_end_point in enumerate(start_end_point_list):
                    # print("each_start_end_point", each_start_end_point)
                    each_start_end_point = each_start_end_point[1:]
                    if flag == len(start_end_point_list) - 1:
                        each_start_end_point = each_start_end_point[:-1]
                    point_list = each_start_end_point.split(", ")
                    # print("point_list",point_list)

                    int_point_list = []
                    for i in point_list:
                        # print(int(i))
                        int_point_list.append(int(i))
                    object_info = [int(object_dict[object_name])] + int_point_list

                    lane_object_list.append(object_info)
                    # print("object_info", object_info)

            if object_serial_num > 11 and object_serial_num <= 16:
                object_info = [int(object_dict[object_name])] + float_list__to_int_list(line[1:])
                # object_info = [int(object_dict[object_name]), float_list__to_int_list(line[1:])]
                # print("object_info", object_info)
                traffic_light_list.append(object_info)


    if len(moveable_object_list) == 0:
        padding_for_moveable_object = [0, 0, 0, 0, 0, 0, 0]
        moveable_object_list.append(padding_for_moveable_object)

    if len(lane_object_list) == 0:
        padding_for_lane_object = [0, 0, 0, 0, 0]
        lane_object_list.append(padding_for_lane_object)

    if len(traffic_light_list) == 0:
        padding_for_traffic_light_object = [0, 0, 0, 0, 0]
        traffic_light_list.append(padding_for_traffic_light_object)

    moveable_object_num_for_an_img = len(moveable_object_list)
    lane_object_num_for_an_img = len(lane_object_list)
    traffic_light_object_num_for_an_img = len(traffic_light_list)



    action_label = None


    with open(action_label_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # intense_loss_line.append(float(line[0]))
            if line[0][:-7] == single_csv_file[:-4]:
                action_label = float_list__to_int_list(line[1:])
        next(reader, None)  # jump to the next line

    # print(action_label)
    # exit()

    possible_moveable_object_list = []
    possible_lane_object_list = []
    possible_traffic_light_list =[]


    for i in one_possible_list:
        # print(i)
        if i < moveable_object_num_for_an_img:
            index_num = i
            possible_moveable_object_list.append(moveable_object_list[index_num])

        if i >= moveable_object_num_for_an_img and i < moveable_object_num_for_an_img + lane_object_num_for_an_img:
            index_num = i - moveable_object_num_for_an_img
            possible_lane_object_list.append(lane_object_list[index_num])

        if i >= lane_object_num_for_an_img + moveable_object_num_for_an_img:
            index_num = i - lane_object_num_for_an_img - moveable_object_num_for_an_img
            possible_traffic_light_list.append(traffic_light_list[index_num])

    if len(possible_moveable_object_list) == 0:
        padding_for_moveable_object = [0, 0, 0, 0, 0, 0, 0]
        possible_moveable_object_list.append(padding_for_moveable_object)

    if len(possible_lane_object_list) == 0:
        padding_for_lane_object = [0, 0, 0, 0, 0]
        possible_lane_object_list.append(padding_for_lane_object)

    if len(possible_traffic_light_list) == 0:
        padding_for_traffic_light_object = [0, 0, 0, 0, 0]
        possible_traffic_light_list.append(padding_for_traffic_light_object)


    three_kinds_of_object_num_in_an_img = len(possible_moveable_object_list), len(possible_lane_object_list), len(possible_traffic_light_list)
    # all_objects_info = (possible_moveable_object_list, possible_lane_object_list, possible_traffic_light_list)
    all_objects_info = all_object_order(possible_moveable_object_list, possible_lane_object_list, possible_traffic_light_list)

    # print("action_label", action_label)
    # exit()
    # print("three_kinds_of_object_num_in_an_img",three_kinds_of_object_num_in_an_img)
    # exit()
    return self_speed_info, all_objects_info, action_label, three_kinds_of_object_num_in_an_img


def all_object_order(moveable_object_list, lane_object_list, traffic_light_list):
    moveable_object_list = moveable_object_list_order(moveable_object_list)
    lane_object_list = lane_object_list_order(lane_object_list)
    traffic_light_list = traffic_object_list_order(traffic_light_list)
    return (moveable_object_list, lane_object_list, traffic_light_list)

def moveable_object_list_order(moveable_object_list):

    object_size_list = []
    for i in moveable_object_list:
        object_size = i[3] * i[4]
        object_size_list.append(object_size)

    sorted_size_id = sorted(range(len(object_size_list)), key = lambda  k: object_size_list[k], reverse = True)

    new_moveable_object_list = []
    for i in sorted_size_id:
        new_moveable_object_list.append(moveable_object_list[i])

    return new_moveable_object_list

def lane_object_list_order(lane_object_list):

    line_length_list = []
    for i in lane_object_list:
        start_point_x = i[1]
        start_point_y = i[2]
        end_point_x = i[3]
        end_point_y = i[4]

        line_length =  (start_point_x - end_point_x) * (start_point_x - end_point_x) + (start_point_y - end_point_y) * (start_point_y - end_point_y)
        line_length_list.append(line_length)

    sorted_size_id = sorted(range(len(line_length_list)), key = lambda  k: line_length_list[k], reverse = True)

    new_line_length_list = []
    for i in sorted_size_id:
        new_line_length_list.append(lane_object_list[i])

    return new_line_length_list

def traffic_object_list_order(traffic_light_list):

    object_size_list = []
    for i in traffic_light_list:
        object_size = i[3] * i[4]
        object_size_list.append(object_size)

    sorted_size_id = sorted(range(len(object_size_list)), key = lambda  k: object_size_list[k], reverse = True)

    new_traffic_light_list = []
    for i in sorted_size_id:
        new_traffic_light_list.append(traffic_light_list[i])

    return new_traffic_light_list

def _to_tensor(device, single_img):

    self_speed_tensor_info = single_img[0]

    moveable_object_tensor_info = single_img[1][0]
    lane_tensor_info = single_img[1][1]
    traffic_light_tensor_info = single_img[1][2]

    action_label_tensor_info = single_img[2]

    moveable_object_num_for_an_img = single_img[3][0]
    lane_object_num_for_an_img = single_img[3][1]
    traffic_light_object_num_for_an_img = single_img[3][2]
    object_num_list = [moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img]

    self_speed_tensor_info = torch.Tensor(self_speed_tensor_info).to(device)

    # exit()
    # print("moveable_object_tensor_info_list", moveable_object_tensor_info_list)

    moveable_object_tensor_info = torch.Tensor(moveable_object_tensor_info).to(device)

    # print("lane_tensor_info_list", lane_tensor_info_list)
    lane_tensor_info = torch.Tensor(lane_tensor_info).to(device)
    traffic_light_tensor_info = torch.Tensor(traffic_light_tensor_info).to(device)
    action_label_tensor_info = torch.Tensor(action_label_tensor_info).to(device)

    # print(moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img)
    # exit()
    all_objects_info = (moveable_object_tensor_info, lane_tensor_info, traffic_light_tensor_info)

    # exit()
    moveable_object_num_for_an_img = object_num_list[0]
    lane_object_num_for_an_img = object_num_list[1]
    traffic_light_object_num_for_an_img = object_num_list[2]
    # print(moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img)
    # exit()
    all_objects_info = (moveable_object_tensor_info, lane_tensor_info, traffic_light_tensor_info)
    three_kinds_of_object_num_in_an_img = (moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img)

    # , lane_tensor_info_list, traffic_light_info_list), action_label_tensor_list
    one_img_tensor_info = (self_speed_tensor_info, all_objects_info, action_label_tensor_info, three_kinds_of_object_num_in_an_img)
    # print("three_kinds_of_object_num_in_an_img", three_kinds_of_object_num_in_an_img)
    return one_img_tensor_info

def single_test(model, one_object_info):
    model.eval()

    with torch.no_grad():

        (self_speed_info, all_objects_info, action_label, three_kinds_of_object_num_in_an_img) = one_object_info

        outputs, object_attention_weights = model(self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img)
        predict_action = torch.sigmoid(outputs) > 0.5

    return predict_action


class Model(nn.Module):
    def __init__(self, config):
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
        # over_all_input_size = 256 + 10

        self.over_all_fc = nn.Linear(over_all_input_size, self.middle_fc_feature1)
        self.middle_fc1 = nn.Linear(self.middle_fc_feature1, self.middle_fc_feature2)
        self.middle_fc2 = nn.Linear(self.middle_fc_feature2, config.num_classes)

    def forward(self, self_speed_info, all_objects_info, three_kinds_of_object_num_in_an_img):
        moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img = three_kinds_of_object_num_in_an_img
        moveable_object_info, lane_object_info, traffic_light_object_info = all_objects_info

        # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img)
        # print("moveable_object_info", moveable_object_info.shape)
        # exit()
        self_speed_tensor_info_list = self_speed_info

        # out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]

        self_speed_middle_info = self.self_speed_fc(self_speed_tensor_info_list)

        def changeable_input_lstm_out(moveable_object_num_for_an_img, moveable_object_info,
                                      moveable_object_lstm):
            # print("moveable_object_num_for_an_img",moveable_object_num_for_an_img)
            moveable_object_info = moveable_object_info.unsqueeze(0)
            moveable_object_out, (moveable_object_out_hn, _) = moveable_object_lstm(moveable_object_info)

            return moveable_object_out

        moveable_object_out = changeable_input_lstm_out(moveable_object_num_for_an_img, moveable_object_info,
                                                        self.moveable_object_lstm, )
        # print("moveable_object_out", moveable_object_out.shape)
        lane_object_out = changeable_input_lstm_out(lane_object_num_for_an_img, lane_object_info,
                                                    self.lane_info_lstm)
        traffic_light_out = changeable_input_lstm_out(traffic_light_object_num_for_an_img,
                                                      traffic_light_object_info, self.traffic_info_lstm)

        # exit()
        def each_kind_object_in_a_img(moveable_object_num_for_an_img, moveable_object_out):
            a_img_moveable_object = moveable_object_out[0,
                                    :int(moveable_object_num_for_an_img), :]
            a_img_moveable_object_importance = torch.matmul(a_img_moveable_object, self.moveable_object_weight)

            return a_img_moveable_object, a_img_moveable_object_importance

        def attentioned_for_a_img(moveable_object_num_for_an_img, moveable_object_out, lane_object_num_for_an_img,
                                  lane_object_out, traffic_light_object_num_for_an_img, traffic_light_out):
            all_object_alpha_in_a_batch = []
            a_img_moveable_object, a_img_moveable_object_importance = each_kind_object_in_a_img(
                moveable_object_num_for_an_img,
                moveable_object_out)
            a_img_lane_object, a_img_lane_object_importance = each_kind_object_in_a_img(lane_object_num_for_an_img,
                                                                                        lane_object_out)
            a_img_traffic_light_object, a_img_traffic_light_object_importance = each_kind_object_in_a_img(
                traffic_light_object_num_for_an_img,
                traffic_light_out)

            all_object_importance = torch.cat((a_img_moveable_object_importance, a_img_lane_object_importance,
                                               a_img_traffic_light_object_importance), dim=0)

            all_object_out = torch.cat(
                (a_img_moveable_object, a_img_lane_object, a_img_traffic_light_object),
                dim=0)
            # print(all_object_importance.shape)
            # exit()
            a_img_all_object_alpha = F.softmax(all_object_importance, 0).unsqueeze(-1)  # [50, x, 1]

            # a_img_all_object_alpha = all_object_importance.unsqueeze(-1)  # [50, x, 1]
            # a_img_all_object_alpha = F.relu(a_img_all_object_alpha)
            a_img_all_object_attentioned = all_object_out * a_img_all_object_alpha

            out_in_a_batch = torch.sum(a_img_all_object_attentioned, 0)  # [128, 256]
            # print(out_in_a_batch.shape)
            out_in_a_batch = out_in_a_batch.unsqueeze(0)

            return out_in_a_batch, a_img_all_object_alpha

        out_in_a_batch, all_object_alpha_in_a_batch = attentioned_for_a_img(moveable_object_num_for_an_img,
                                                                            moveable_object_out,
                                                                            lane_object_num_for_an_img,
                                                                            lane_object_out,
                                                                            traffic_light_object_num_for_an_img,
                                                                            traffic_light_out)

        # print("out_in_a_batch.shape, all_object_alpha_in_a_batch.shape",out_in_a_batch.shape, all_object_alpha_in_a_batch.shape)
        # exit()
        self_speed_middle_info = self_speed_middle_info.unsqueeze(0)
        # print(self_speed_middle_info.shape, out_in_a_batch.shape)
        # exit()
        # object_info_plus_self_speed = torch.cat((self_speed_middle_info, out_in_a_batch), 1)
        object_info_plus_self_speed = out_in_a_batch
        middle_fc_feature = self.drop(self.relu1(self.over_all_fc(object_info_plus_self_speed)))
        output = self.middle_fc1(middle_fc_feature)
        # output = self.middle_fc0(output)
        output = self.middle_fc2(output)

        return output, all_object_alpha_in_a_batch


def read_csv_file(file_path):
    samples = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            samples.append(row)
    return samples
def find_original_attention(attention_csv_path):
    attention_samples = read_csv_file(attention_csv_path)

    # for i in attention_samples:
    #     print(i)

    predicted_action = attention_samples[0]
    actual_action = attention_samples[1]
    moveable_object_num, lane_object_num, traffic_light_object_num = attention_samples[2]

    moveable_object_num = int(moveable_object_num)
    lane_object_num = int(lane_object_num)
    traffic_light_object_num = int(traffic_light_object_num)

    maxum_moveable_num = moveable_object_num
    maxum_lane_num = lane_object_num
    maxum_traffic_light_num = traffic_light_object_num
    # exit()
    moveable_objects_alpha_index = 3
    moveable_objects_alpha = attention_samples[
                             moveable_objects_alpha_index:moveable_objects_alpha_index + maxum_moveable_num]


    lane_alpha_index = moveable_objects_alpha_index + maxum_moveable_num + maxum_moveable_num
    lane_alpha = attention_samples[lane_alpha_index:lane_alpha_index + maxum_lane_num]

    traffic_light_alpha_index = lane_alpha_index + maxum_lane_num + maxum_lane_num
    traffic_light_alpha = attention_samples[
                          traffic_light_alpha_index:traffic_light_alpha_index + maxum_traffic_light_num]

    all_object_alpha = moveable_objects_alpha + lane_alpha + traffic_light_alpha
    return all_object_alpha, predicted_action, actual_action

def calcu_a_img_MEE(single_img_name, single_img_folder_path): # minumum effective explanation



    single_img_path = os.path.join(single_img_folder_path, single_img_name)

    action_label_csv_path = "action_label/zhang_pesudo_img.csv"
    object_settings_path = "settings/object_setting_no_repeat.csv"

    # attention_csv_folder_path = 'TextRNN_Att_3RNN_1attention_changeable_input(new_lane_plus_sp)'
    attention_csv_folder_path = "TextRNN_Att_3RNN_1attention_changeable_input(BTS_new_lane)"
    save_csv_folder_path = os.path.join("attention_csv_folder", attention_csv_folder_path)
    save_csv_folder_path = os.path.join(save_csv_folder_path, single_img_name)

    all_object_alpha, predicted_action, actual_action = find_original_attention(save_csv_folder_path)
    predicted_action_boole = []
    for i in predicted_action:
        if i == "True":
            predicted_action_boole.append(True)
        if i == "False":
            predicted_action_boole.append(False)

    all_object_alpha_index = sorted(range(len(all_object_alpha)), key=lambda k: all_object_alpha[k])

    from_big_to_small_index = []

    for i in range(len(all_object_alpha_index) - 1, -1, -1):
        from_big_to_small_index.append(all_object_alpha_index[i])

    one_possible_list = []
    counter_flag = 0
    # print(len(all_object_alpha))
    # exit()
    for i in from_big_to_small_index:
        counter_flag = counter_flag + 1
        one_possible_list.append(i)

        self_speed_info, all_objects_info, action_label, three_kinds_of_object_num_in_an_img = one_img_csv_process(
            single_img_path, single_img_name,
            object_settings_path, action_label_csv_path, one_possible_list)

        one_img_info = self_speed_info, all_objects_info, action_label, three_kinds_of_object_num_in_an_img
        one_img_tensor_info = _to_tensor(device, one_img_info)
        test_prediction = single_test(model, one_img_tensor_info)

        test_prediction = str(test_prediction.cpu().numpy())

        test_prediction = test_prediction.split(" ")
        test_prediction_boole = []
        for j in test_prediction:
            if j != "":
                # print(j)
                if j[0] == "T":
                    test_prediction_boole.append(True)
                if j[0] == "F":
                    test_prediction_boole.append(False)
        # EMM_percentage = 1
        if predicted_action_boole == test_prediction_boole:
            print("all object number: ", len(from_big_to_small_index), " counter_flag: ", counter_flag)
            EMM_percentage = float( counter_flag / len(from_big_to_small_index) )
            break
    return EMM_percentage, all_objects_info, three_kinds_of_object_num_in_an_img

def save_csv_file(save_csv_path, all_objects_info, three_kinds_of_object_num_in_an_img):
    list = []

    with open(save_csv_path, "a", newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')            # write the result to the new csv
        writer.writerow(three_kinds_of_object_num_in_an_img)
        for i in all_objects_info:
            for j in i:
                writer.writerow(j)







if __name__ == '__main__':


    # np.random.seed(1)  # 让numpy的随机数出来的东西固定
    # torch.manual_seed(1)  # 让CPU生成固定的随机数
    # torch.cuda.manual_seed_all(1)  # 让GPU生成固定的随机数
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样 cuda的随机数种子

    # 以上四个随机数种子的设置是我们可以复现出我们的结果，像是一个combo，来保证每一次的训练结果，正确率相同

    # print(one_img_info)
    # exit()
    train_img_csv_folder_path = "train_pesudo_img_label_821"
    vali_img_csv_folder_path = "vali_pesudo_img_label_821"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = "TextRNN_Att_3RNN_1attention_changeable_input(new_lane_plus_sp)"
    model_name = "TextRNN_Att_3RNN_1attention_changeable_input(BTS_new_lane)"
    trained_model_folder_path = "trained_model"
    model_path = os.path.join(trained_model_folder_path, model_name + '.ckpt')
    config = Config()

    model = Model(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    train_file_name_list = os.listdir(train_img_csv_folder_path)
    vali_file_name_list = os.listdir(vali_img_csv_folder_path)

    train_EMM_percentage_list= []
    vali_EMM_percentage_list = []

    attention_MEE_folder_name = "attention_MEE"


    attention_list = []
    img_name_list = []
    for single_img_name in train_file_name_list:

        EMM_percentage, all_objects_info, three_kinds_of_object_num_in_an_img = calcu_a_img_MEE(single_img_name, train_img_csv_folder_path)
        train_EMM_percentage_list.append(EMM_percentage)
        save_csv_path = os.path.join(attention_MEE_folder_name, single_img_name)
        save_csv_file(save_csv_path, all_objects_info, three_kinds_of_object_num_in_an_img)


    for single_img_name in vali_file_name_list:

        EMM_percentage, all_objects_info, three_kinds_of_object_num_in_an_img = calcu_a_img_MEE(single_img_name, vali_img_csv_folder_path)
        vali_EMM_percentage_list.append(EMM_percentage)
        save_csv_path = os.path.join(attention_MEE_folder_name, single_img_name)
        save_csv_file(save_csv_path, all_objects_info, three_kinds_of_object_num_in_an_img)



    print("train_EMM_percentage_list, mean", sum(train_EMM_percentage_list) / len(train_EMM_percentage_list))
    print("vali_EMM_percentage_list, mean", sum(vali_EMM_percentage_list) / len(vali_EMM_percentage_list))

