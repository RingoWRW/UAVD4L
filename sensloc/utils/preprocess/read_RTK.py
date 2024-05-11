import numpy as np
import os
import glob
import pandas as pd
import argparse

def read_RTK_csv(input_RTK_path, output_csv_path, start_time, end_time):

    RTK_list = []
    RTK_list += glob.glob(input_RTK_path + "*.csv")

    lat_list = []
    lon_list = []
    H_list = []
    time_list = []

    s_hour = start_time.split(' ')[:2]
    s_minute = start_time.split(' ')[3:5]
    s_second = start_time.split(' ')[6:8]

    e_hour = end_time.split(' ')[:2]
    e_minute = end_time.split(' ')[3:5]
    e_second = end_time.split(' ')[6:8]

 
    for i in range(len(RTK_list)):
        ground_truth_information = pd.read_csv(RTK_list[i], encoding = 'gb2312')
        ground_truth_information_time = ground_truth_information["日期时间"]
        ground_lat = ground_truth_information['lat']
        ground_lon = ground_truth_information['lon']
        ground_H = ground_truth_information['H']
        ground_time = ground_truth_information['日期时间']

        gs_hour = ground_truth_information_time[0].split(' ')[:2]
        gs_minute = ground_truth_information_time[0].split(' ')[3:5]
        gs_second = ground_truth_information_time[0].split(' ')[6:8]

        
        for k in range(len(ground_truth_information_time)):
            gs_hour = ground_truth_information_time[k].split(' ')[:2]
            gs_minute = ground_truth_information_time[k].split(' ')[3:5]
            gs_second = ground_truth_information_time[k].split(' ')[6:8]

            if s_hour <= gs_hour <= e_hour:
                if s_minute <= gs_minute <= e_minute:
                    if s_second <= gs_second <= e_second:

                        lat_list.append(ground_lat[k])
                        lon_list.append(ground_lon[k])
                        H_list.append(ground_H[k])
                        time_list.append(ground_time[k])

         

    dataframe = pd.DataFrame({ 'time':time_list,'lat':lat_list, 'lon':lon_list, 'H':H_list})
    name = start_time[:10] + '_' + start_time[11:13] + '_' + start_time[14:16] + '_' + start_time[17:20] + '.csv'

    dataframe.to_csv(output_csv_path + '/' + name, index=False, sep=',')

def main():
    parser = argparse.ArgumentParser(description="Read RTK file to csv.")
    parser.add_argument("--input_RTK_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/query/video/RTK/")
    parser.add_argument("--output_csv_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/query/txt/RTK/")
    parser.add_argument("--start_time", default="2023-03-29 10:37:57")
    parser.add_argument("--end_time", default="2023-03-29 11:22:00")
    args = parser.parse_args()

    read_RTK_csv(args.input_RTK_path, args.output_csv_path, args.start_time, args.end_time)


if __name__ == "__main__":

    main()
                             