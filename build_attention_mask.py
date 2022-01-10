import os
import json
import cv2
from PIL import Image
import numpy
import csv
import show_info_data_gyro
import numpy as np
import numpy.matlib
import math

def grid_info_to_box_info(lane_info):
    position_info_dict_list = []
    for each_grid in lane_info:
        grid_serial_num = each_grid[1]
        img_width = 1280
        img_height = 720
        grid_x_num = 16
        grid_y_num = 9
        grid_width = img_width / grid_x_num
        grid_height = img_height / grid_y_num
        # print("grid_serial_num", grid_serial_num)
        grid_serial_num = grid_serial_num - 1
        y_grid_num = grid_serial_num // grid_x_num
        x_grid_num = grid_serial_num - y_grid_num * grid_x_num
        # print("x_grid_num", y_grid_num, x_grid_num)

        center_point_x = x_grid_num * (img_width / grid_x_num) + float(grid_width / 2)
        center_point_y = y_grid_num * (img_height / grid_y_num) + float(grid_height / 2)

        position_info_x1 = center_point_x - grid_width / 2
        position_info_y1 = center_point_y - grid_height / 2
        position_info_x2 = center_point_x + grid_width / 2
        position_info_y2 = center_point_y + grid_height / 2

        position_info_dict = {'x1': position_info_x1, 'y1': position_info_y1, 'x2': position_info_x2,
                              'y2': position_info_y2}
        position_info_dict_list.append(position_info_dict)


    return position_info_dict_list


def moveable_object_info_translate(moveable_objects_info):
    save_for_future_moveable_position_list = []
    for each_moveable_object in moveable_objects_info:
        position_info_x1 = each_moveable_object[1] - each_moveable_object[3] / 2
        position_info_y1 = each_moveable_object[2] - each_moveable_object[4] / 2
        position_info_x2 = each_moveable_object[1] + each_moveable_object[3] / 2
        position_info_y2 = each_moveable_object[2] + each_moveable_object[4] / 2

        position_info_dict = {'x1': position_info_x1, 'y1': position_info_y1, 'x2': position_info_x2, 'y2': position_info_y2}
        save_for_future_moveable_position_list.append(position_info_dict)

    return save_for_future_moveable_position_list

def traffic_light_info_translate(traffic_light_info):
    traffic_light_position_info_list = []
    for each_traffic_light in traffic_light_info:
        position_info_x1 = each_traffic_light[1] - each_traffic_light[3] / 2
        position_info_y1 = each_traffic_light[2] - each_traffic_light[4] / 2
        position_info_x2 = each_traffic_light[1] + each_traffic_light[3] / 2
        position_info_y2 = each_traffic_light[2] + each_traffic_light[4] / 2

        position_info_dict = {'x1': position_info_x1, 'y1': position_info_y1, 'x2': position_info_x2, 'y2': position_info_y2}
        traffic_light_position_info_list.append(position_info_dict)

    return traffic_light_position_info_list

def superimpose_picture(original_img_j, attention_img):
    rescaled_act_img_j = attention_img - np.amin(attention_img)
    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed_original_img_j = 0 * original_img_j + 1 * heatmap * 255
    return overlayed_original_img_j


def csv_info_str_data_to_num(moveable_objects_info):
    output_moveable_objects_info = []
    for i in moveable_objects_info:
        i = str(i)
        i = i[3:]
        i = i[:-3]
        str_list = i.split(",")
        float_i_list = []
        for str_i in str_list:
            float_i = float(str_i)
            float_i_list.append(float_i)

        output_moveable_objects_info.append(float_i_list)

    return output_moveable_objects_info

def csv_attention_str_data_to_num(moveable_objects_info):
    output_moveable_objects_info = []
    for i in moveable_objects_info:
        i = str(i)
        i = i[3:]
        i = i[:-3]
        float_i = float(i)
        output_moveable_objects_info.append(float_i)
    # print(output_moveable_objects_info)
    return output_moveable_objects_info

def make_attention_with_each_box_info(object_alpha, object_box_info, attention_map):

    horizontal_line_start = int(object_box_info["x1"])
    horizontal_line_end = int(object_box_info["x2"])

    longitudinal_line_start = int(object_box_info["y1"])
    longitudinal_line_end = int(object_box_info["y2"])

    for horizontal_line in range(horizontal_line_start, horizontal_line_end):
        for longitudinal_line in range(longitudinal_line_start, longitudinal_line_end):
            attention_map[longitudinal_line, horizontal_line] = object_alpha

    return attention_map

def float_list_to_int_list(float_list):
    int_list = []
    for i in float_list:
        int_list.append(int(i))
    return int_list

def make_attention_with_each_start_end_point(object_alpha, object_box_info, attention_map):

    (x1, y1) = object_box_info[1], object_box_info[2]
    (x2, y2) = object_box_info[3], object_box_info[4]
    gap_length = 10
    line_length = math.sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) )
    gap_number = int( line_length / gap_length )
    if gap_number % 2 == 0:
        gap_number = gap_number + 1

    # gap_number = 5 # 至少5个, 只能为奇数 2n + 1
    x_gap_length = (x2 - x1) / gap_number
    y_gap_length = (y2 - y1) / gap_number
    each_line_list = []
    for i in range(gap_number):
        if i % 2 == 0:
            each_line = []
            # print(i)
            x_temp = x1 + i * x_gap_length
            y_temp = y1 + i * y_gap_length

            point = []
            point.append(x_temp)
            point.append(y_temp)
            point = float_list_to_int_list(point)
            each_line.append(point)

            point = []
            point.append(x_temp + x_gap_length)
            point.append(y_temp + y_gap_length)
            point = float_list_to_int_list(point)
            each_line.append(point)

            each_line_list.append(each_line)
    # print(each_line_list)
    # exit()
    for each_point in each_line_list:
        # print(each_point)
        center_point_x = int((each_point[0][0] + each_point[1][0]) / 2)
        center_point_y = int((each_point[0][1] + each_point[1][1]) / 2)

        box_length = abs(each_point[0][0] - each_point[1][0])
        box_height = abs(each_point[0][1] - each_point[1][1])

        x1 = center_point_x - int(box_length / 2)
        x2 = center_point_x + int(box_length / 2)
        y1 = center_point_y - int(box_height / 2)
        y2 = center_point_y + int(box_height / 2)

        object_box_info = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        # print(object_box_info)
        # exit()
        attention_map = make_attention_with_each_box_info(object_alpha, object_box_info, attention_map)
    return attention_map


def make_attention_with_box_info(objects_alpha, objects_box_info, attention_map):
    if len(objects_alpha) != len(objects_box_info):
        print("len(objects_alpha) != len(objects_box_info)")
        exit()
    for serial_num in range(len(objects_alpha)):
        # print(objects_box_info[serial_num])
        # exit()
        attention_map = make_attention_with_each_box_info(objects_alpha[serial_num], objects_box_info[serial_num], attention_map)
    return attention_map

def make_attention_with_start_end_point(objects_alpha, objects_box_info, attention_map):
    if len(objects_alpha) != len(objects_box_info):
        print("len(objects_alpha) != len(objects_box_info)")
        exit()
    for serial_num in range(len(objects_alpha)):
        print(objects_alpha[serial_num], objects_box_info[serial_num])
        # exit()
        attention_map = make_attention_with_each_start_end_point(objects_alpha[serial_num], objects_box_info[serial_num], attention_map)
    # exit()
    return attention_map

def make_a_attention_mask(attention_csv_folder, wanted_test_name, target_img_folder_path, original_img_folder_name):
    original_img_name = wanted_test_name + "_52.jpg"
    original_img_path = os.path.join(original_img_folder_name, original_img_name)

    attention_csv_path = os.path.join(attention_csv_folder, wanted_test_name + ".csv")
    attention_samples = read_csv_file(attention_csv_path)

    print(wanted_test_name)
    # for i in attention_samples:
    #     print(i)


    maxum_moveable_num = 31
    maxum_lane_num = 115
    maxum_traffic_light_num = 11

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
    moveable_objects_alpha = attention_samples[moveable_objects_alpha_index:moveable_objects_alpha_index + maxum_moveable_num]
    moveable_objects_info = attention_samples[ moveable_objects_alpha_index + maxum_moveable_num : moveable_objects_alpha_index + maxum_moveable_num + maxum_moveable_num]


    lane_alpha_index = moveable_objects_alpha_index + maxum_moveable_num + maxum_moveable_num
    lane_alpha = attention_samples[lane_alpha_index:lane_alpha_index + maxum_lane_num]
    lane_info = attention_samples[lane_alpha_index + maxum_lane_num:lane_alpha_index + maxum_lane_num + maxum_lane_num]

    traffic_light_alpha_index = lane_alpha_index + maxum_lane_num + maxum_lane_num
    traffic_light_alpha = attention_samples[traffic_light_alpha_index:traffic_light_alpha_index + maxum_traffic_light_num]
    traffic_light_info = attention_samples[traffic_light_alpha_index + maxum_traffic_light_num:traffic_light_alpha_index + maxum_traffic_light_num + maxum_traffic_light_num]

    moveable_objects_info = csv_info_str_data_to_num(moveable_objects_info)
    moveable_objects_alpha = csv_attention_str_data_to_num(moveable_objects_alpha)
    # print("moveable_object_num, lane_object_num, traffic_light_object_num", moveable_object_num, lane_object_num,
    #       traffic_light_object_num)
    # print(len(moveable_objects_info))
    # print(len(moveable_objects_alpha))
    # print(len(lane_info))
    # print(len(lane_alpha))
    # print(len(traffic_light_info))
    # print(len(traffic_light_alpha))
    # exit()
    lane_info = csv_info_str_data_to_num(lane_info)
    lane_alpha = csv_attention_str_data_to_num(lane_alpha)

    traffic_light_info = csv_info_str_data_to_num(traffic_light_info)
    traffic_light_alpha = csv_attention_str_data_to_num(traffic_light_alpha)


    moveable_objects_alpha, moveable_objects_info = pick_eligiable_object_and_attention(moveable_objects_alpha, moveable_objects_info)
    moveable_objects_box_info = moveable_object_info_translate(moveable_objects_info)

    lane_alpha, lane_info = pick_eligiable_object_and_attention(lane_alpha,lane_info)
    # print(lane_alpha, lane_info)
    # exit()
    # lane_box_info = grid_info_to_box_info(lane_info)

    traffic_light_alpha, traffic_light_info = pick_eligiable_object_and_attention(traffic_light_alpha, traffic_light_info)
    traffic_light_box_info = traffic_light_info_translate(traffic_light_info)

    original_image = cv2.imread(original_img_path)

    attention_map = np.zeros((720, 1280))
    # print(image.shape, image[0].shape)
    # print(moveable_objects_box_info)
    # exit()
    attention_map = make_attention_with_box_info(moveable_objects_alpha, moveable_objects_box_info, attention_map)
    # print(lane_alpha, lane_info)
    # exit()
    attention_map = make_attention_with_start_end_point(lane_alpha, lane_info, attention_map)

    attention_map = make_attention_with_box_info(traffic_light_alpha, traffic_light_box_info, attention_map)

    if len(lane_alpha) != 0:
        lane_alpha_alpha_max_single_img = max(lane_alpha)
    else:
        lane_alpha_alpha_max_single_img = 0

    attentioned_original_img = superimpose_picture(original_image, attention_map)

    attentioned_img_name = wanted_test_name + "_attentioned" + ".jpg"
    attentioned_img_path = os.path.join(target_img_folder_path, attentioned_img_name)
    cv2.imwrite( attentioned_img_path , attentioned_original_img)

    return lane_alpha_alpha_max_single_img





def pick_eligiable_object_and_attention(moveable_objects_alpha, moveable_objects_info):
    output_moveable_objects_alpha = []
    output_moveable_objects_info = []
    eligiable_object_num = 0
    for i in moveable_objects_info:
        if i[0] == 0:
            break
        else:
            output_moveable_objects_info.append(i)
            eligiable_object_num = eligiable_object_num + 1

    for i_flag, i in enumerate( moveable_objects_alpha):
        if i_flag >=eligiable_object_num:
            break
        else:
            output_moveable_objects_alpha.append(i)

    # print(len(output_moveable_objects_alpha))
    # print(len(output_moveable_objects_info))

    return output_moveable_objects_alpha, output_moveable_objects_info

def read_csv_file(file_path):
    samples = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            samples.append(row)
    return samples
if __name__ == "__main__":

    target_img_path = os.path.abspath('.')
    pesudo_img_folder_name = "Pesudo_img_folder_500"
    # attentioned_img_folder_name = "attentioned_img_folder"
    attentioned_img_folder_name = "TextRNN_Att_3RNN_1attention_changeable_input_img_nostfmx"
    # attentioned_img_folder_name = "each_category_attentioned_img_folder"

    # target_img_path = os.path.join(target_img_path, pesudo_img_folder_name)
    target_folder_path = os.path.join(target_img_path, attentioned_img_folder_name)

    white_img_name = "white_img.jpg"
    # attention_csv_folder = "TextRNN_Att"
    attention_csv_folder = "TextRNN_Att_3RNN_1attention_changeable_input_nosftmx"
    file_name_list = os.listdir(attention_csv_folder)
    required_name_list = []
    for i in file_name_list:
        required_name = i[:-4]
        required_name_list.append(required_name)

    lane_alpha_max = 0
    for required_name in required_name_list:
        lane_alpha_alpha_max_single_img = make_a_attention_mask(attention_csv_folder, required_name, target_folder_path, pesudo_img_folder_name)
        # if lane_alpha_alpha_max_single_img > lane_alpha_max:
        #     lane_alpha_max = lane_alpha_alpha_max_single_img
        #     lane_alpha_max_imgname = required_name
    # print("required_name", lane_alpha_max, lane_alpha_max_imgname)