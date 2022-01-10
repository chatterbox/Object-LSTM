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


def float_list__to_int_list(float_list):
    new_list = []
    for i in float_list:
        new_list.append(int(i))
    return new_list

def make_symmed_moveable_object(object_info):
    new_object_info = object_info
    symmed_num = calcu_symmetric_num(object_info[1])
    new_object_info[1] = symmed_num
    new_object_info[-2] = -object_info[-2]
    return new_object_info

def make_symmed_lane_object(object_info):

    new_object_info = object_info
    symmed_num = calcu_symmetric_lane_num(object_info[1])
    new_object_info[1] = symmed_num
    return new_object_info

def make_symmed_traffic_light_object(object_info):
    new_object_info = object_info
    symmed_num = calcu_symmetric_num(object_info[1])
    new_object_info[1] = symmed_num
    return new_object_info

def calcu_symmetric_num(input_num, img_size_x = 1280):

    symmed_num = int(img_size_x / 2) - (input_num - int(img_size_x / 2))
    return symmed_num

def calcu_symmetric_lane_num(input_num):

    column_num = input_num // 16
    input_num = input_num % 16
    if input_num == 0 and column_num >= 1:
        column_num = column_num - 1
        input_num = 16
    first_list = []
    second_list = []
    for i in range(8, 0, -1):
        first_list.append(i)
    for i in range(9, 17):
        second_list.append(i)

    if input_num in first_list:
        correspond_index = first_list.index(input_num)
        correspond_num = second_list[correspond_index]
    else:
        correspond_index = second_list.index(input_num)
        correspond_num = first_list[correspond_index]
    correspond_num = correspond_num + column_num * 16
    return correspond_num

def make_symmed_img(single_csv_path, object_settings_path):
    object_dict = {}
    with open(object_settings_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # intense_loss_line.append(float(line[0]))
            object_dict[line[0]] = line[1]
        next(reader, None)  # jump to the next line

    symmed_moveable_object_list = []
    symmed_lane_object_list = []
    symmed_traffic_light_list = []
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
                object_info = make_symmed_moveable_object(object_info)
                symmed_moveable_object_list.append(object_info)
                # print("object_info", object_info)

            if object_serial_num > 7 and object_serial_num <= 11:
                grid_string = line[1]
                grid_string = grid_string[1:]
                grid_string = grid_string[:-1]
                grid_list = grid_string.split(",")

                for each_grid in grid_list:
                    object_info = [int(object_dict[object_name]), int(each_grid)]
                    object_info = make_symmed_lane_object(object_info)
                    symmed_lane_object_list.append(object_info)
                    # print("object_info", object_info)

            if object_serial_num > 11 and object_serial_num <= 16:
                object_info = [int(object_dict[object_name])] + float_list__to_int_list(line[1:])
                # object_info = [int(object_dict[object_name]), float_list__to_int_list(line[1:])]
                # print("object_info", object_info)
                object_info = make_symmed_traffic_light_object(object_info)
                symmed_traffic_light_list.append(object_info)

        make_symmed_csv_file(self_speed_info, symmed_moveable_object_list, symmed_lane_object_list, symmed_traffic_light_list)

def make_symmed_csv_file(self_speed_info, symmed_moveable_object_list, symmed_lane_object_list, symmed_traffic_light_list):
    pass

def get_unsymmed_csv_path_list(train_folder_path):
    file_name_list = os.listdir(train_folder_path)
    unsymmed_csv_path_list = []
    for file_name in file_name_list:
        single_csv_path = os.path.join(train_folder_path, file_name)
        unsymmed_csv_path_list.append(single_csv_path)
    return unsymmed_csv_path_list

if __name__ == '__main__':


    object_settings_path = "settings/object_setting_no_repeat.csv"
    # config = x.Config(dataset, embedding)
    action_label_csv_path = "action_label/zhang_pesudo_img.csv"
    train_folder_path = "train_pesudo_img_label"
    vali_folder_path = "vali_pesudo_img_label"

    unsymmed_csv_path_list = get_unsymmed_csv_path_list(train_folder_path)
    for unsymmed_csv_path in unsymmed_csv_path_list:
        make_symmed_img(unsymmed_csv_path, object_settings_path)

    # 以上四个随机数种子的设置是我们可以复现出我们的结果，像是一个combo，来保证每一次的训练结果，正确率相同


