import csv
import sys
from math import sqrt
from math import sin
from math import atan2
import pandas as pd
import time
import os

sys.setrecursionlimit(1000000)

# colors for terminal
CRED = '\033[91m'
CREDEND = '\033[0m'
CGREY = '\x1b[6;30;42m'
CGREYEND = '\x1b[0m'

def main():
    currentDirectory = os.getcwd()
    # dynamicPath = os.path.dirname(currentDirectory)
    dynamicPath = currentDirectory + '/Compressed_datasets/DeadR/'
    try:
        os.makedirs(dynamicPath)
    except OSError:
        print ("Creation of the directory %s failed" % dynamicPath)
    else:
        print ("Successfully created the directory %s" % dynamicPath)

    input_file = sys.argv[1]
    save_file = dynamicPath+'DeadReckoning_AVG.csv'

    data = pd.read_csv(input_file).sort_values(['mmsi', 'TIMESTAMP'])
    data.to_csv(input_file, index=False)

    size = data.groupby('mmsi').size().agg(list)
    distance = []  # list to keep the average distance for each ship

    with open(input_file, 'r') as fin:
        reader = csv.reader(fin)
        next(reader)

        avg, d, count = 0, 0, 0
        row2 = next(reader)
        try:
            for idx, next_row in enumerate(reader):
                row1, row2 = row2, next_row

                if row1[5] != row2[5]:
                    print('For MMSI', row1[5], 'the initial points in the dataset are: ', count + 1)
                    distance.append(avg)
                    avg, d, count = 0, 0, 0
                    row1, row2 = row2, next(reader)

                    if row1[5] != row2[5]:
                        distance.append(0)
                        print('For MMSI', row1[5], 'the initial points in the dataset are:  1')
                        idx += 1
        except:
            pass

            count += 1
            d = calculate_d(row1, row2)
            avg = avg + (d - avg) / count

    print('For MMSI', row1[5], 'the initial points in the dataset are: ', count + 1)
    distance.append(avg)

    dead_reckoning(input_file, save_file, distance, size)

def dead_reckoning(input_file, save_file, threshold, initial_points):
    with open(input_file, 'r') as fin, open(save_file, 'w') as fout:
        reader = csv.reader(fin)
        row1 = next(reader)

        writer = csv.writer(fout)
        writer.writerow(row1)

        d = []
        a = []

        start_idx = 0
        k = 1
        max_d = 0
        idx_of_thr = 0

        row2 = next(reader)
        writer.writerow(row2)
        compressed_points = 1
        total_compressed_points = 0

        check = ''
        try:
            for next_row in reader:
                previous, row1, row2 = row1, row2, next_row

                if row1[5] != row2[5]:

                    k = 1
                    writer.writerow(row1)
                    compressed_points += 1
                    total_compressed_points += compressed_points

                    print('\n', 'MMSI: ', row1[5], '--Coordinate pairs after compression: ', compressed_points, ' Initial points', initial_points[idx_of_thr])
                    print('\n', (1 - compressed_points / initial_points[idx_of_thr]) * 100, 'percent compressed')
                    print('------------------------------------')

                    compressed_points = 0
                    writer.writerow(row2)
                    compressed_points += 1
                    max_d = 0
                    idx_of_thr += 1

                    row1, row2 = row2, next(reader)

                    if row1[5] != row2[5]:
                        total_compressed_points += compressed_points

                        print('\n', 'MMSI: ', row1[5], '--Coordinate pairs after compression: ', compressed_points, ' Initial points', initial_points[idx_of_thr])
                        print('\n', (1 - compressed_points / initial_points[idx_of_thr]) * 100, 'percent compressed')
                        print('------------------------------------')

                        idx_of_thr += 1
                        row1, row2 = row2, next(reader)
                        writer.writerow(row1)
                        compressed_points = 1
                        check = row1
                        continue
                    check = row1    # a check which need to take place later in order to not write the same row twice
                    continue
        except:
            pass
            distance = calculate_d(row1, row2)
            angle = calculate_angle(row1, row2)
            d.append(distance)
            a.append(angle)

            max_d = abs(d[k - 1] * sin(a[k - 1] - a[start_idx]))
            if max_d > threshold[idx_of_thr]:
                max_d = 0
                if previous != check:
                    writer.writerow(previous)
                    compressed_points += 1
                start_idx = k - 1
            k += 1
        writer.writerow(row2)
        compressed_points += 1
        total_compressed_points += compressed_points

        print('\n', 'MMSI: ', row1[5], '--Coordinate pairs after compression: ', compressed_points, ' Initial points', initial_points[idx_of_thr])
        print('\n', (1 - compressed_points / initial_points[idx_of_thr]) * 100, 'percent compressed')
        print('------------------------------------')

        total_initial_points = sum(i for i in initial_points)
        print("\n***************************************************************")
        print(CRED + "                  Aggregated results" + CREDEND)
        print("***************************************************************\n")
        print(CGREY + "Unique MMSI: " + CGREYEND, len(initial_points))
        print(CGREY + "Initial points: " + CGREYEND, total_initial_points)

        if total_compressed_points:
            print(CGREY + "Points after compression: " + CGREYEND, total_compressed_points)
        try:
            if total_compressed_points:
                print(CGREY + "Compression rate: " + CGREYEND, (1 - total_compressed_points / total_initial_points) * 100)
        except ZeroDivisionError:
            pass


def calculate_d(x, y):
    # print (x, y)
    return sqrt((float(y[1]) - float(x[1])) ** 2 + (float(y[0]) - float(x[0])) ** 2)

def calculate_angle(x, y):
    lat_diff = float(y[1]) - float(x[1])
    lon_diff = float(y[0]) - float(x[0])
    return atan2(lon_diff, lat_diff)


start_time = time.time()
if __name__ == '__main__':
    print('                        ------------------------------------------------------------------------------------------------------------')
    print('                        ----------------------------------------- Dead Reckoning - Algorithm ---------------------------------------')
    print('                        ------------------------------------------------------------------------------------------------------------')
    main()
print(CGREY + 'Total time elapsed (in seconds) ' + CGREYEND, str(time.time() - start_time))
