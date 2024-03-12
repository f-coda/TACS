import csv
import os
import sys
from math import sqrt
from pointClass import Point_ref
from pointClass import Point_write_to_csv
import pandas as pd
import time

sys.setrecursionlimit(1000000)

# colors for terminal
CRED = '\033[91m'
CREDEND = '\033[0m'
CGREY = '\x1b[6;30;42m'
CGREYEND = '\x1b[0m'


# load all elements from dataset
def load_data_from_file(provided_dataset):
    # load the point data
    df = pd.read_csv(provided_dataset)
    df['LATITUDE-LONGITUDE-COG-SOG-TIMESTAMP-mmsi'] = list(
        map(list, zip(df['LATITUDE'], df['LONGITUDE'], df['COG'], df['SOG'], df['TIMESTAMP'],df['mmsi'])))
    grouped = df.groupby('mmsi')['LATITUDE-LONGITUDE-COG-SOG-TIMESTAMP-mmsi'].apply(list)
    res_ufc = [[k, v] for k, v in grouped.items()]
    return res_ufc


# Euclidean distance, between two points
# Input: Two Points
# Output:  Distance (float), in degrees
def distance(p_a, p_b):
    return sqrt((p_a.latitude - p_b.latitude) ** 2 + (p_a.longitude - p_b.longitude) ** 2)

# Distance from a point to a line that formed by two points
# Input: a Point, start_point + end_point -> line points
# Output: distance to line (float), in degrees
def point_line_Distance(point, start_point, end_point):
    if start_point == end_point:
        return distance(point, start_point)
    else:
        un_dist = abs(
            (end_point.latitude - start_point.latitude) * (start_point.longitude - point.longitude) - (start_point.latitude - point.latitude) * (
                        end_point.longitude - start_point.longitude)
        )
        # Euclidean distance, between two points
        n_dist = sqrt(
            (end_point.latitude - start_point.latitude) ** 2 + (end_point.longitude - start_point.longitude) ** 2
        )
        if n_dist == 0:
            return 0
        else:
            return un_dist / n_dist


# gather the compressed points inside a list in order to sum up and print results
comp_points = []
def gather_compressed_pointsLen_in_each_run(comppoi):
    comp_points.append(comppoi)
    return comp_points

# variable that used to separate the parsing loop
first_run = 0
def convert_array_to_pointobj(points, unique_mmsi, numberofinitialpoints,dynamicPath):

    # split the array
    obj_points = []
    array_points = [c for c in points[0]]

    # convert to Point obj
    for i in range(len(array_points)):
        obj_points.append(Point_write_to_csv(array_points[i][0], array_points[i][1], array_points[i][2], array_points[i][3],
                                array_points[i][4],array_points[i][5]))

    define_thresholds_for_distance = define_distance_threshold(obj_points)
    try:
        # Call the rdp dynamic function with the data in the correct form thus Point obj
        objects_dynamic_rdp = rdp_dynamic(obj_points, define_thresholds_for_distance)
        raw_data = []
        global first_run

        # If first run then create the dataframe with the column headers
        if first_run == 0:
            for i in range(len(objects_dynamic_rdp)):
                print (objects_dynamic_rdp[i].mmsi)
                raw_data.append({
                    'latitude': objects_dynamic_rdp[i].latitude,
                    'longitude': objects_dynamic_rdp[i].longitude,
                    'cog': objects_dynamic_rdp[i].cog,
                    'sog': objects_dynamic_rdp[i].sog,
                    'timestamp': objects_dynamic_rdp[i].timestamp,
                    'mmsi': objects_dynamic_rdp[i].mmsi
                })
                df = pd.DataFrame(raw_data,
                                  columns=['latitude', 'longitude', 'cog', 'sog', 'timestamp',
                                           'mmsi'])

                first_run = first_run + 1
                #print dynamicPath+'/Compression/Compressed_datasets/DP/Douglas-Peucker_AVG.csv'
            df.to_csv(dynamicPath+'Douglas-Peucker_AVG.csv', mode='a', index=False)
        # if this is NOT the first run, create the dataframe WITHOUT the column headers
        else:
            for i in range(len(objects_dynamic_rdp)):
                raw_data.append({

                    'latitude': objects_dynamic_rdp[i].latitude,
                    'longitude': objects_dynamic_rdp[i].longitude,
                    'cog': objects_dynamic_rdp[i].cog,
                    'sog': objects_dynamic_rdp[i].sog,
                    'timestamp': objects_dynamic_rdp[i].timestamp,
                    'mmsi': objects_dynamic_rdp[i].mmsi,

                })
                df = pd.DataFrame(raw_data,
                                  columns=['latitude', 'longitude', 'cog', 'sog', 'timestamp',
                                           'mmsi'])
            df.to_csv(dynamicPath+'Douglas-Peucker_AVG.csv', mode='a', index=False, header=False)

        print (" \n MMSI: ", unique_mmsi, "--Coordinate pairs after compression: ", len(
            objects_dynamic_rdp), " Initial points: ", numberofinitialpoints)
        print ("\n", (1 - float(len(objects_dynamic_rdp)) / float(numberofinitialpoints)) * 100, 'percent compressed')
        print ("------------------------------------")
        list_with_compressed_points_in_len = gather_compressed_pointsLen_in_each_run(len(objects_dynamic_rdp))
        # return the length of compressed points, sum up the results
        return sum(list_with_compressed_points_in_len)
    except:
        pass

# function to compute the average distance BUT in degrees because DP works with degrees
def point_line_Distance_thres(point, start_point, end_point):
    if start_point == end_point:
        return distance(point, start_point)
    else:
        un_dist = abs(
            (end_point.latitude - start_point.latitude) * (start_point.longitude - point.longitude) - (start_point.latitude - point.latitude) * (
                        end_point.longitude - start_point.longitude)
        )
        # Euclidean distance, between two points
        n_dist = sqrt(
            (end_point.latitude - start_point.latitude) ** 2 + (end_point.longitude - start_point.longitude) ** 2
        )
        if n_dist == 0:
            return 0
        else:
            return un_dist / n_dist

# define distance threshold dynamically
def define_distance_threshold(points):
    list_with_distances = []
    if len(points) <= 2:
        return points
    else:
        for i in range(0, len(points) - 1):
            line_from_point_distance = point_line_Distance_thres(points[i], points[0], points[-1])
            list_with_distances.append(line_from_point_distance)
        average = sum(list_with_distances) / float(len(list_with_distances))
        return float(average)

def rdp_dynamic(obj_points, epsilon):
    max_distance = 0.0
    index_value = 0
    for i in range(1, len(obj_points) - 1):
        # find the distance points[i] from the line between the start ad end point
        line_from_point_distance = point_line_Distance(obj_points[i], obj_points[0], obj_points[-1])
        # iterate until max distance is found from the point examined and the line
        if line_from_point_distance > max_distance:
            index_value = i
            max_distance = line_from_point_distance

    # if max distance from point is greater than the epsilon (user define threshold)
    # split the Object that contains the points in this point and keep this point with the max distance and examine the two subsegments again
    if max_distance > epsilon:
        return rdp_dynamic(obj_points[:index_value + 1], epsilon)[:-1] + rdp_dynamic(
            obj_points[index_value:], epsilon)
    # if max distance from point is lesser than epsilon, discard the examined point
    else:
        return [obj_points[0], obj_points[-1]]


if __name__ == "__main__":
    print ("\n")
    print ("                        ------------------------------------------------------------------------------------------------------------")
    print ("                        ---------------------------------------Douglas-Peucker  Algorithm -----------------------------------")
    print ("                        ------------------------------------------------------------------------------------------------------------")
    print ("\n")

    currentDirectory = os.getcwd()
    #dynamicPath = os.path.dirname(currentDirectory)
    dynamicPath = currentDirectory + '/Compressed_datasets/DP/'
    try:
        os.makedirs(dynamicPath)
    except OSError:
        print ("Creation of the directory %s failed" % dynamicPath)
    else:
        print ("Successfully created the directory %s" % dynamicPath)

    fetch_ids_and_pairs_combined = load_data_from_file(sys.argv[1])

    length_of_initial_points = []
    for i in range(len(fetch_ids_and_pairs_combined)):
        length_of_initial_points.append(len(fetch_ids_and_pairs_combined[i][1]))
        print ("For MMSI:", fetch_ids_and_pairs_combined[i][0], "the initial points in dataset are", len(
            fetch_ids_and_pairs_combined[i][1]))
    print ("\n")

    unique_ids = []
    start = time.time()
    for i in range(len(fetch_ids_and_pairs_combined)):
        # add the unique ids in a list to measure the distinct mmsi
        unique_ids.append(fetch_ids_and_pairs_combined[i][0])
        # takes as arguments 1)the coordinates, 2)the mmsi and 3)the number of initial points (these points create the trajectory) in each mmsi
        # 4) the dynamic path which points to current working directory
        compressed_points = convert_array_to_pointobj(([Point_ref(fetch_ids_and_pairs_combined[i][1])]),
                                                      fetch_ids_and_pairs_combined[i][0],
                                                      len(fetch_ids_and_pairs_combined[i][1]),dynamicPath)

    # ------------------------------------------------------------------------------------------------------------------
    print ("\n***************************************************************")
    print           (CRED + "                  Aggregated results" + CREDEND)
    print ("***************************************************************\n")

    print ("Unique MMSI:", len(unique_ids))
    print ("Initial points:", sum(length_of_initial_points))
    if compressed_points:
        print ("Points after compression:", compressed_points)
    try:
        if compressed_points:
            print ("Compression rate:",(1 - float((compressed_points)) / float(sum(length_of_initial_points))) * 100)
    except ZeroDivisionError:
        pass
    end = time.time()
    print ("Total time elapsed (in seconds):", (end - start))



