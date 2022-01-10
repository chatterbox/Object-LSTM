# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import csv
import random
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


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

def build_dataset(train_folder_path, vali_folder_path, object_settings_path, action_label_csv_path, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum, object_order_flag, random_arange_flag, shuffle_num):
    # print(vocab)
    # exit()
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

    def calcu_symmetric_num(input_num, img_size_x=1280):

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

    def padding_object(shuffleed_all_objects_info, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum):
        moveable_object_list, lane_object_list, traffic_light_list = shuffleed_all_objects_info
        for i in range(len(moveable_object_list), max_moveable_object_maxnum):
            padding_for_moveable_object = [0, 0, 0, 0, 0, 0, 0]
            moveable_object_list.append(padding_for_moveable_object)

        for i in range(len(lane_object_list), max_lane_object_maxnum):
            padding_for_lane_object = [0, 0, 0, 0, 0]
            lane_object_list.append(padding_for_lane_object)

        for i in range(len(traffic_light_list), max_traffic_light_maxnum):
            padding_for_traffic_light_object = [0, 0, 0, 0, 0]
            traffic_light_list.append(padding_for_traffic_light_object)
        padded_shuffleed_all_objects_info = moveable_object_list, lane_object_list, traffic_light_list
        return padded_shuffleed_all_objects_info

    def random_arrange_object(moveable_object_list, lane_object_list, traffic_light_list, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum, one_img_info, shuffle_num):
        one_img_ramdom_arrange_info_array = []
        ramdom_arrange_info = []

        # shuffle_num = 5

        for i in range(shuffle_num):
            exec("moveable_object_list_temp{} = {}".format(i, moveable_object_list))
            exec("lane_object_list{} = {}".format(i, lane_object_list))
            exec("traffic_light_list_temp{} = {}".format(i, traffic_light_list))

            exec('random.shuffle(moveable_object_list_temp{})'.format(i))
            exec('random.shuffle(lane_object_list{})'.format(i))
            exec('random.shuffle(traffic_light_list_temp{})'.format(i))

            loc = locals()
            exec('shuffleed_all_objects_info = (moveable_object_list_temp{}, lane_object_list{}, traffic_light_list_temp{})'.format(i,i,i))
            shuffleed_all_objects_info = loc["shuffleed_all_objects_info"]




            shuffleed_all_objects_info = padding_object(shuffleed_all_objects_info, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum)
            ramdom_arrange_info = (one_img_info[0], shuffleed_all_objects_info, one_img_info[2], one_img_info[3], one_img_info[4])
            one_img_ramdom_arrange_info_array.append(ramdom_arrange_info)


        return one_img_ramdom_arrange_info_array

    def load_dataset(folder_path, object_settings_path, object_order_flag):
        file_name_list = os.listdir(folder_path)
        object_dict = {}
        with open(object_settings_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                # intense_loss_line.append(float(line[0]))
                object_dict[line[0]] = line[1]
            next(reader, None)  # jump to the next line
        # print(object_dict)
        # exit()
        all_imgs_contents = []
        for single_csv_file in file_name_list:
            single_csv_path = os.path.join(folder_path, single_csv_file)
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
                        for flag, each_start_end_point in enumerate(start_end_point_list) :
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
                    # intense_loss_line.append(float(line[0]))
                # print(moveable_object_list)
                # print(lane_object_list)
                # print(traffic_light_list)

            # print(len(moveable_object_list), len(lane_object_list), len(traffic_light_list))



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

            three_kinds_of_object_num_in_an_img = (moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img)

            one_img_info = (
            self_speed_info, 0, action_label, single_csv_file, three_kinds_of_object_num_in_an_img)

            if random_arange_flag == True:
                one_img_ramdom_arrange_info_array = random_arrange_object(moveable_object_list, lane_object_list, traffic_light_list, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum, one_img_info, shuffle_num)
                all_imgs_contents = all_imgs_contents + one_img_ramdom_arrange_info_array
            else:
                if object_order_flag == True:
                    all_objects_info = all_object_order(moveable_object_list,
                                                                  lane_object_list,
                                                                  traffic_light_list)
                else:
                    all_objects_info = (moveable_object_list, lane_object_list, traffic_light_list)

                all_objects_info = padding_object(all_objects_info, max_moveable_object_maxnum, max_lane_object_maxnum, max_traffic_light_maxnum)
                one_img_info = (
                    self_speed_info, all_objects_info, action_label, single_csv_file, three_kinds_of_object_num_in_an_img)
                all_imgs_contents.append(one_img_info)


        return all_imgs_contents

    def load_symmed_dataset(folder_path, object_settings_path, pad_size=32):
        file_name_list = os.listdir(folder_path)
        object_dict = {}
        with open(object_settings_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                # intense_loss_line.append(float(line[0]))
                object_dict[line[0]] = line[1]
            next(reader, None)  # jump to the next line
        # print(object_dict)
        # exit()
        all_imgs_contents = []
        for single_csv_file in file_name_list:
            single_csv_path = os.path.join(folder_path, single_csv_file)
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
                        object_info = make_symmed_moveable_object(object_info)
                        moveable_object_list.append(object_info)
                        # print("object_info", object_info)

                    if object_serial_num > 7 and object_serial_num <= 11:
                        grid_string = line[1]
                        grid_string = grid_string[1:]
                        grid_string = grid_string[:-1]
                        grid_list = grid_string.split(",")

                        for each_grid in grid_list:
                            object_info = [int(object_dict[object_name]), int(each_grid)]
                            object_info = make_symmed_lane_object(object_info)
                            lane_object_list.append(object_info)
                            # print("object_info", object_info)

                    if object_serial_num > 11 and object_serial_num <= 16:
                        object_info = [int(object_dict[object_name])] + float_list__to_int_list(line[1:])
                        # object_info = [int(object_dict[object_name]), float_list__to_int_list(line[1:])]
                        # print("object_info", object_info)
                        object_info = make_symmed_traffic_light_object(object_info)
                        traffic_light_list.append(object_info)
                    # intense_loss_line.append(float(line[0]))
                # print(moveable_object_list)
                # print(lane_object_list)
                # print(traffic_light_list)

            # print(len(moveable_object_list), len(lane_object_list), len(traffic_light_list))
            for i in range(len(moveable_object_list), max_moveable_object_maxnum):
                padding_for_moveable_object = [0, 0, 0, 0, 0, 0, 0]
                moveable_object_list.append(padding_for_moveable_object)

            for i in range(len(lane_object_list), max_lane_object_maxnum):
                padding_for_lane_object = [0, 0]
                lane_object_list.append(padding_for_lane_object)

            for i in range(len(traffic_light_list), max_traffic_light_maxnum):
                padding_for_traffic_light_object = [0, 0, 0, 0, 0]
                traffic_light_list.append(padding_for_traffic_light_object)

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

            one_img_info = (self_speed_info, moveable_object_list, lane_object_list, traffic_light_list, action_label, single_csv_file)
            if len(moveable_object_list) != 31:
                print("moveable_object_list padding error", len(moveable_object_list))
                exit()
            if len(lane_object_list) != 115:
                print("lane_object_list padding error", len(lane_object_list))
                exit()
            if len(traffic_light_list) != 11:
                print("lane_object_list padding error", len(traffic_light_list))
                exit()
            all_imgs_contents.append(one_img_info)
        return all_imgs_contents

    train = load_dataset(train_folder_path, object_settings_path, object_order_flag)
    vali = load_dataset(vali_folder_path, object_settings_path, object_order_flag)
    # symmed_train = load_symmed_dataset(train_folder_path, object_settings_path)
    # symmed_vali = load_symmed_dataset(vali_folder_path, object_settings_path)

    return train, vali


def int_list_to_tensor(int_list, device):
    tensor_list = torch.Tensor(int_list).to(device)
    return  tensor_list





class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device


    def _to_tensor(self, datas):

        # print(len(datas))
        # print(len(datas[0]))

        self_speed_tensor_info_list = []
        moveable_object_tensor_info_list = []
        lane_tensor_info_list = []
        traffic_light_tensor_info_list = []
        action_label_tensor_info_list = []
        img_name_list = []
        moveable_object_num_for_an_img_list = []
        lane_object_num_for_an_img_list = []
        traffic_light_object_num_for_an_img_list = []
        for single_img in datas:
            # print(single_img[3])
            # print(len(single_img[0]))
            # print(len(single_img[0][0]), single_img[0][0])
            self_speed_tensor_info_list.append(single_img[0])

            moveable_object_tensor_info_list.append(single_img[1][0])
            lane_tensor_info_list.append(single_img[1][1])
            traffic_light_tensor_info_list.append(single_img[1][2])

            action_label_tensor_info_list.append(single_img[2])
            img_name_list.append(single_img[3])

            moveable_object_num_for_an_img_list.append(single_img[4][0])
            lane_object_num_for_an_img_list.append(single_img[4][1])
            traffic_light_object_num_for_an_img_list.append(single_img[4][2])



        # print(len(moveable_object_tensor_info_list), len(moveable_object_tensor_info_list[0]), len(moveable_object_tensor_info_list[0][0]))
        # moveable_object_tensor_info_list = np.array(moveable_object_tensor_info_list)
        # print(moveable_object_tensor_info_list.size)
        # exit()
        self_speed_tensor_info_list = torch.Tensor(self_speed_tensor_info_list).to(self.device)
        # print("moveable_object_tensor_info_list", moveable_object_tensor_info_list)
        moveable_object_tensor_info_list = torch.Tensor(moveable_object_tensor_info_list).to(self.device)

        # print("lane_tensor_info_list", lane_tensor_info_list)
        lane_tensor_info_list = torch.Tensor(lane_tensor_info_list).to(self.device)
        traffic_light_tensor_info_list = torch.Tensor(traffic_light_tensor_info_list).to(self.device)

        action_label_tensor_info_list = torch.Tensor(action_label_tensor_info_list).to(self.device)

        moveable_object_num_for_an_img_list = torch.Tensor(moveable_object_num_for_an_img_list).to(self.device)
        lane_object_num_for_an_img_list = torch.Tensor(lane_object_num_for_an_img_list).to(self.device)
        traffic_light_object_num_for_an_img_list = torch.Tensor(traffic_light_object_num_for_an_img_list).to(self.device)

        # print(self_speed_tensor_info_list.shape)
        # print(moveable_object_tensor_info_list.shape)
        # print(lane_tensor_info_list.shape)
        # print(traffic_light_tensor_info_list.shape)
        # print(action_label_tensor_info_list.shape)
        all_objects_info = (moveable_object_tensor_info_list, lane_tensor_info_list, traffic_light_tensor_info_list)
        three_kinds_of_object_num_in_an_img = (moveable_object_num_for_an_img_list, lane_object_num_for_an_img_list, traffic_light_object_num_for_an_img_list)
        # exit()
        # , lane_tensor_info_list, traffic_light_info_list), action_label_tensor_list
        return self_speed_tensor_info_list, all_objects_info, action_label_tensor_info_list, img_name_list, three_kinds_of_object_num_in_an_img

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, batch_size, device):
    iter = DatasetIterater(dataset, batch_size, device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
