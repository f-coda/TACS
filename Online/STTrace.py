import sys
import csv
import datetime
import math
import os
import time
from math import atan2
from math import sqrt
import pandas as pd
import geopy.distance

sys.setrecursionlimit(1000000)
# colours
CRED = '\033[91m'
CREDEND = '\033[0m'
CGREY = '\x1b[6;30;42m'
CGREYEND = '\x1b[0m'


def main():

    currentDirectory = os.getcwd()

    dynamicPath = currentDirectory + '/Compressed_datasets/STTrace/'
    try:
        os.makedirs(dynamicPath)
    except OSError:
        print('Creation of the directory %s failed' % dynamicPath)
    else:
        print('Successfully created the directory %s' % dynamicPath)

    input_file = sys.argv[1]
    save_file = dynamicPath + 'STTrace_AVG.csv'

    data = pd.read_csv(input_file).sort_values(['mmsi', 'TIMESTAMP'])
    length_of_initial_points = len(data)
    data.to_csv(input_file, index=False)
    grouped = data.groupby('mmsi')

    size = grouped.size()
    for mmsi, sz in size.items():
        print('For MMSI:', mmsi, 'the initial points in the dataset are', sz)
    size = size.agg(list)

    speed_threshold = find_average_speed(input_file)
    orientation_threshold = find_average_ori(input_file)

    with open(input_file, 'r') as source:
        reader = csv.reader(source)
        header = next(reader)

        idx = 0
        compressed_points = []
        total_compressed_points = 0
        row2, row3 = next(reader), next(reader)

        compressed_points.append(row2)
        compressed_points.append(row3)
        row4 = next(reader)
        maximum_size = 10 * size[idx] // 100

        first_time = True
        for next_row in reader:
            row1, row2, row3, row4 = row2, row3, row4, next_row

            if row3[5] != row4[5]:
                if compressed_points[len(compressed_points)-1] != row3:
                    compressed_points.append(row3)

                print('MMSI: ', row3[5], '--Coordinate pairs after compression: ', len(compressed_points), 'Initial points: ', size[idx])
                print('\n', (1 - len(compressed_points) / size[idx]) * 100, 'percent compressed')
                print('-------------------------------------')

                if first_time:
                    compressed_points.insert(0, header)
                    first_time = False
                write_to_file(compressed_points, save_file)
                total_compressed_points += len(compressed_points)
                compressed_points.clear()

                idx += 1
                maximum_size = 10 * size[idx] // 100

                row2, row3, row4 = row4, next(reader), next(reader)
                compressed_points.append(row2)
                compressed_points.append(row3)
                continue

            if row1[5] != row2[5]:
                compressed_points.pop(len(compressed_points) - 1)
                print('MMSI: ', row1[5], '--Coordinate pairs after compression: ', len(compressed_points), 'Initial points: ', size[idx])
                print('\n', (1 - len(compressed_points) / size[idx]) * 100, 'percent compressed')
                print('-------------------------------------')
                idx += 1
                maximum_size = 10 * size[idx] // 100
                write_to_file(compressed_points, save_file)
                total_compressed_points += len(compressed_points)
                compressed_points.clear()
                compressed_points.append(row2)
                compressed_points.append(row3)
                continue

            if row1[5] != row3[5]:
                print('MMSI: ', row1[5], '--Coordinate pairs after compression: ', len(compressed_points), 'Initial points: ', size[idx])
                print('\n', (1 - len(compressed_points) / size[idx]) * 100, 'percent compressed')
                print('-------------------------------------')
                idx += 1
                maximum_size = 10 * size[idx] // 100
                write_to_file(compressed_points, save_file)
                total_compressed_points += len(compressed_points)
                compressed_points.clear()
                compressed_points.append(row3)
                compressed_points.append(row4)
                row2, row3, row4 = row3, row4, next(reader)
                continue

            if row1[5] != row4[5]:
                compressed_points.pop(len(compressed_points) - 1)
                compressed_points.append(row3)
                print('MMSI: ', row1[5], '--Coordinate pairs after compression: ', len(compressed_points), 'Initial points: ', size[idx])
                print('\n', (1 - len(compressed_points) / size[idx]) * 100, 'percent compressed')
                print('-------------------------------------')
                idx += 1
                maximum_size = 10 * size[idx] // 100
                write_to_file(compressed_points, save_file)
                total_compressed_points += len(compressed_points)
                compressed_points.clear()
                compressed_points.append(row3)
                compressed_points.append(row4)
                row2, row3 = row4, next(reader)
                continue

            result = safe_areas(compressed_points[len(compressed_points) - 1], compressed_points[len(compressed_points) - 2], row1, row2, row3, speed_threshold[idx], orientation_threshold[idx])

            if result:
                continue
            else:
                if len(compressed_points) >= maximum_size:
                    sttrace(compressed_points, row3, row4)
                else:
                    compressed_points.append(row3)


    print('MMSI: ', row1[5], '--Coordinate pairs after compression: ', len(compressed_points), 'Initial points: ', size[idx])
    print('\n', (1 - len(compressed_points) / size[idx]) * 100, 'percent compressed')
    print('-------------------------------------')
    write_to_file(compressed_points, save_file)
    total_compressed_points += len(compressed_points) - 1

    print("\n***************************************************************")
    print(CRED + "                  Aggregated results             " + CREDEND)
    print("***************************************************************\n")

    print(CGREY + "Unique MMSI: " + CGREYEND, len(size))
    print(CGREY + "Initial points: " + CGREYEND, length_of_initial_points)
    if total_compressed_points:
        print (CGREY + "Points after compression: " + CGREYEND, total_compressed_points)
    try:
        if compressed_points:
            print(CGREY + "Compression rate: " + CGREYEND, (1 - total_compressed_points / length_of_initial_points) * 100)
    except ZeroDivisionError:
        pass

def find_average_speed(dataset):
    try:
        with open(dataset, 'r') as source:
            reader = csv.reader(source)
            next(reader)

            avg_speed = []

            first_row = next(reader)
            temp_mmsi = first_row[5]

            for row in reader:
                if row[5] != temp_mmsi:
                    if first_row != last_row:
                        avg_speed.append(calculate_distance(first_row, last_row) / time_traveled(first_row, last_row))
                    else:
                        avg_speed.append(0)

                    temp_mmsi = row[5]
                    first_row = row

                last_row = row
    except:
        pass
    avg_speed.append(calculate_distance(first_row, last_row) / time_traveled(first_row, last_row))
    return avg_speed

def find_average_ori(dataset):
    try:
        with open(dataset, 'r') as source:
            reader = csv.reader(source)

            avg_ori = []
            avg = 0
            row2 = next(reader)
            row3 = next(reader)

            i = 1

            for next_row in reader:
                row1, row2, row3 = row2, row3, next_row

                if row2[1] != row3[1]:

                    avg_ori.append(avg)
                    avg = 0
                    i = 1
                    continue


                if row1[1] != row2[1]:
                    continue

                i += 1
                a = getAngle(row1, row2, row3)
                avg += (a - avg) / i
    except:
        pass
    avg_ori.append(avg)
    return avg_ori


# LAW OF COSINES
def getAngle(first, second, third):
    ang = math.degrees(math.atan2(float(third[0]) - float(second[0]), float(third[1]) - float(second[1])) - math.atan2(float(first[0]) - float(second[0]), float(first[1]) - float(second[1])))
    return ang + 360 if ang < 0 else ang


# real earth distance in meters
def calculate_distance(first, second):
    origin = (first[1], first[0])
    dist = (second[1], second[0])
    return geopy.distance.distance(origin, dist).meters

# total time traveled between two points
def time_traveled(first_row, last_row):

    date1 = first_row[4]
    date2 = last_row[4]
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date1 = datetime.datetime.timestamp(date1)
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.timestamp(date2)

    if abs(date2 - date1) == 0:
        return 1

    return abs(date2 - date1)

# predicting the two safe areas
def safe_areas(sample_a, sample_b, traj_a, traj_b, traj_c, threshold_speed, threshold_ori):

    x = safe_speed(sample_a, sample_b, traj_a, traj_b, traj_c, threshold_speed)
    y = safe_orientation(sample_a, sample_b, traj_a, traj_b, traj_c, threshold_ori)

    return x and y


def safe_speed(sample_a, sample_b, point_a, point_b, point_c, speed_threshold):

    sample_speed = calculate_speed(sample_a, sample_b)
    trajectory_speed = calculate_speed(point_a, point_b)
    current_speed = (calculate_speed(point_b, point_c))

    if abs( sample_speed - current_speed ) > speed_threshold or abs( trajectory_speed - current_speed ) > speed_threshold:
        return False
    else:
        return True


def calculate_speed(a, b):
    return calculate_distance(a, b) / time_traveled(a, b)


def safe_orientation( sample_a, sample_b, point_a, point_b, point_c, ori_threshold):

    angle_sample_ab = calculate_angle(sample_a, sample_b)
    angle_bc = calculate_angle(point_b, point_c)
    angle_sample_ab_bc = angle_bc - angle_sample_ab
    angle_trajectory_ab = calculate_angle(point_a, point_b)
    angle_trajectory_ab_bc = angle_bc - angle_trajectory_ab


    if abs(angle_sample_ab_bc) > ori_threshold or abs(angle_trajectory_ab_bc) > ori_threshold:
        return False
    else:
        return True


def calculate_angle(a, b):

    lat_diff = float(b[0]) - float(a[0])
    lon_diff = float(b[1]) - float(a[1])

    return atan2(lon_diff, lat_diff)


def sttrace(buffer, point, next_point):
    min_sed = 1000000
    min_id = -1
    row1 = buffer[0]
    row2 = buffer[1]
    sed = 0
    try:
        for idx, i in enumerate(buffer[2:]):
            sed = calculate_sed(row1, row2, i)
            if sed < min_sed:
                min_sed = sed
                min_id = idx + 1

        new_sed = calculate_sed(buffer[len(buffer) - 1], point, next_point)

        if new_sed < sed:
            buffer.pop(min_id)
            buffer.append(point)
    except:
        pass


def calculate_sed(first, second, third):
    return sqrt((float(second[1]) - (float(first[1]) + float(third[1])) / 2) ** 2 + (float(second[0]) - (float(first[0]) + float(third[0])) / 2) ** 2)


def write_to_file(buffer, save_file):
    with open(save_file, 'a') as dest:
        writer = csv.writer(dest)

        for line in buffer:
            writer.writerow(line)

start_time = time.time()
if __name__ == '__main__':
    print()
    print('                        -----------------------------------------------------------------------------------------------')
    print('                        --------------------------------------- STTRace - ALGORITHM -----------------------------------')
    print('                        -----------------------------------------------------------------------------------------------')
    print()
    main()
print(CGREY + "Total time elapsed (in seconds): " + CGREYEND, str(time.time() - start_time))
