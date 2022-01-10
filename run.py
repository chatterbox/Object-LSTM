# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import TextRNN
import TextRNN_Att
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import os
import csv
from importlib import import_module

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

if __name__ == '__main__':
    
    # config = x.Config(dataset, embedding)
    action_label_csv_path = "action_label/zhang_pesudo_img.csv"
    train_folder_path = "train_pesudo_img_label_821"
    vali_folder_path = "vali_pesudo_img_label_821"
    object_settings_path = "settings/object_setting_no_repeat.csv"
    np.random.seed(1) # 让numpy的随机数出来的东西固定
    torch.manual_seed(1)  # 让CPU生成固定的随机数
    torch.cuda.manual_seed_all(1)   # 让GPU生成固定的随机数
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样 cuda的随机数种子

    # 以上四个随机数种子的设置是我们可以复现出我们的结果，像是一个combo，来保证每一次的训练结果，正确率相同

    start_time = time.time()
    print("Loading data...")

    max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum = find_padding_size_for_each_info(train_folder_path, object_settings_path)

    vali_moveable_object_maxnum, vali_lane_object_maxnum, vali_traffic_light_maxnum = find_padding_size_for_each_info(
        vali_folder_path, object_settings_path)


    if vali_moveable_object_maxnum > max_moveable_object_maxnum:
        max_moveable_object_maxnum = vali_moveable_object_maxnum
    if vali_lane_object_maxnum > max_lane_object_maxnum:
        max_lane_object_maxnum = vali_lane_object_maxnum
    if vali_traffic_light_maxnum > max_traffic_light_maxnum:
        max_traffic_light_maxnum = vali_traffic_light_maxnum

    print(max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum)

    random_arange_flag = True

    self_speed_flag = True
    object_order_flag = False
    if random_arange_flag == True:
        object_order_flag = False
    shuffle_num = 25
    train_data, vali_data = build_dataset(train_folder_path, vali_folder_path, object_settings_path, action_label_csv_path, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum, object_order_flag, random_arange_flag, shuffle_num)


    print(len(train_data), len(vali_data))
    # exit()
    # print(train_data[0][0])
    # print(train_data[0][1])
    # print(train_data[0][2])
    # print(train_data[0][3])
    #
    # print("*"*100)



    batch_size = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # exit()
    # print(len(train_data), train_data[0])
    # exit()
    train_iter = build_iterator(train_data, batch_size, device) # 返回一个打包好的训练数据集，batch size为128
    # exit()
    # print(train_iter[0])
    # counter = 0
    # for i in train_iter:
    #     print(i[0][1].size(), i[0][0].size(), i[1].size()) # torch.Size([128]) torch.Size([128, 32]) torch.Size([128])
    #     counter = counter + 1
        # exit()
    # print(counter)
    # exit()
    vali_iter = build_iterator(vali_data, batch_size, device)
    # test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)

    print("Time usage:", time_dif)

    # train
    model_name = "TextRNN_Att_3RNN_1attention_test_selfspd"
    # model_name = "TextRNN_Att_3RNN_1attention"
    # model_name = "TextRNN_Att"
    # model_name = "TextRNN"
    x = import_module(model_name)

    config = x.Config()
    model = x.Model(config, self_speed_flag).to(device)

    print(model.parameters)
    train(model, train_iter, vali_iter, config, self_speed_flag)
    # exit()
    # attention_csv_file_path = "/home/zhang/Desktop/zhang/Doctor research/Build criterion driver model/attention_csv_folder/TextRNN_Att_3RNN_1attention/00f0dd0f-5e9c9557.csv"
    # samples = read_csv_file(attention_csv_file_path)
    #
    # attention_sum = 0
    # for n_flag, i in enumerate(samples):
    #     if n_flag >= 2 and n_flag < (2+max_moveable_object_maxnum) or n_flag >= 64 and n_flag < (64+max_lane_object_maxnum) or n_flag >= 294 and n_flag < (294+max_traffic_light_maxnum):
    #         i = str(i)
    #         attention_num = i[3:]
    #         attention_num = float(attention_num[:-3])
    #         attention_sum = attention_sum + attention_num
    # print("attention_sum", attention_sum)
    # exit()

    random_arange_flag = False
    train_data, vali_data = build_dataset(train_folder_path, vali_folder_path, object_settings_path,
                                          action_label_csv_path, max_moveable_object_maxnum, max_lane_object_maxnum,
                                          max_traffic_light_maxnum, object_order_flag, random_arange_flag, shuffle_num)

    train_iter = build_iterator(train_data, batch_size, device)
    vali_iter = build_iterator(vali_data, batch_size, device)

    model_name = "TextRNN_Att_3RNN_1attention_changeable_input(RA_test_selfspd)"
    # model_name = "TextRNN_Att_3RNN_1attention_changeable_input_nosftmx"

    # model_name = "TextRNN_Att_3RNN_1attention_changeable_input_final"
    trained_model_folder_path =  "trained_model"
    model_path = os.path.join(trained_model_folder_path, model_name + '.ckpt')
    pseudo_img_name = "00f0dd0f-5e9c9557.csv"
    save_csv_folder_path = os.path.join("attention_csv_folder", model_name)

    x.save_RNNattention(save_csv_folder_path, model_path, train_iter, config, device, self_speed_flag)
    x.save_RNNattention(save_csv_folder_path, model_path, vali_iter, config, device, self_speed_flag)
    exit()

    # pseudo_img_path = os.path.join(train_folder_path, pseudo_img_name)
    # predict_action, action_label, moveable_object_alpha, lane_alpha, traffic_light_alpha, moveable_object_info, lane_info, traffic_object_info = x.show_RNNattention(model_path, pseudo_img_name, train_iter, config, device, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum)
    # print("predict_action", predict_action)
    # print("action_label", action_label)
    # print("moveable_object_info", moveable_object_info)
    # print("moveable_object_alpha", moveable_object_alpha)
    #
    # print("lane_info", lane_info)
    # print("lane_alpha", lane_alpha)
    #
    # print("traffic_object_info", traffic_object_info)
    # print("traffic_light_alpha", traffic_light_alpha)
