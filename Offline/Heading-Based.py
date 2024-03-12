import os
import sys
from pointClass import Point_ref
from pointClass import Point_write_to_csv
import pandas as pd
import time
import math

sys.setrecursionlimit(10000)

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
        map(list, zip(df['LATITUDE'], df['LONGITUDE'], df['COG'], df['SOG'], df['TIMESTAMP'], df['mmsi'])))
    grouped = df.groupby('mmsi')['LATITUDE-LONGITUDE-COG-SOG-TIMESTAMP-mmsi'].apply(list)
    res_ufc = [[k, v] for k, v in grouped.items()]
    return res_ufc

# gather the compressed points in list in order to sum up and print results
comp_points = []
def gather_compressed_pointsLen_in_each_run(comppoi):
    comp_points.append(comppoi)
    return comp_points

# variable that used to separate the number of run
first_run = 0

def convert_array_to_pointobj(points, unique_mmsi ,numberofinitialpoints, dynamicPath):
    # split the array
    obj_points = []
    array_points = [c for c in points[0]]

    # convert to Point obj
    for i in range(len(array_points)):
        obj_points.append(
            Point_write_to_csv(array_points[i][0], array_points[i][1], array_points[i][2], array_points[i][3],
                                  array_points[i][4], array_points[i][5]))
    define_thresholds_for_heading = define_heading_threshold(obj_points)

    try:
        # Call the time ratio speed function with the data in the correct form (Point obj)
        objects_dynamic_rdp = heading_based(obj_points,define_thresholds_for_heading)
        raw_data = []
        global first_run

        if first_run == 0:
            for i in range(len(objects_dynamic_rdp)):
                raw_data.append({

                    'latitude': objects_dynamic_rdp[i].latitude,
                    'longitude': objects_dynamic_rdp[i].longitude,
                    'cog': objects_dynamic_rdp[i].cog,
                    'sog': objects_dynamic_rdp[i].sog,
                    'timestamp': objects_dynamic_rdp[i].timestamp,
                    'mmsi': objects_dynamic_rdp[i].mmsi
                })
                df = pd.DataFrame(raw_data, columns=['latitude', 'longitude', 'cog', 'sog', 'timestamp',
                                           'mmsi'])
                first_run = first_run + 1
            df.to_csv(dynamicPath+'Heading-Based_AVG.csv', mode='a', index=False)
        # if this is NOT the first run create the dataframe WITHOUT the column headers
        else:
            for i in range(len(objects_dynamic_rdp)):
                raw_data.append({

                    'latitude': objects_dynamic_rdp[i].latitude,
                    'longitude': objects_dynamic_rdp[i].longitude,
                    'cog': objects_dynamic_rdp[i].cog,
                    'sog': objects_dynamic_rdp[i].sog,
                    'timestamp': objects_dynamic_rdp[i].timestamp,
                    'mmsi': objects_dynamic_rdp[i].mmsi
                })

                df = pd.DataFrame(raw_data, columns=['latitude', 'longitude', 'cog', 'sog', 'timestamp',
                                           'mmsi'])
            df.to_csv(dynamicPath+'Heading-Based_AVG.csv', mode='a', index=False, header=False)

        print (" \n MMSI: ", unique_mmsi, "--Coordinate pairs after compression: ", len(
            objects_dynamic_rdp), " Initial points: ", numberofinitialpoints)

        print ("\n", (1 - float(len(objects_dynamic_rdp)) / float(numberofinitialpoints)) *100, 'percent compressed')
        print ("------------------------------------")
        list_with_compressed_points_in_len = gather_compressed_pointsLen_in_each_run(len(objects_dynamic_rdp))
        return sum(list_with_compressed_points_in_len)
    except:
        pass

# Law of Cosines
def getAngle(a, b, c):
    #latitude/longitude
    ang = math.degrees(math.atan2(c.longitude-b.longitude, c.latitude-b.latitude) - math.atan2(a.longitude-b.longitude, a.latitude-b.latitude))
    return ang + 360 if ang < 0 else ang

# find the threshold for each trajectory
def define_heading_threshold(points):
    list_with_headings = []
    if len(points) <= 2:
        return points
    else:
        for i in range(1, len(points)-1):
            list_with_headings.append(getAngle( points[i-1],  points[i],  points[i+1]))
        average = sum(list_with_headings) / float(len(list_with_headings))
        return average

def heading_based(points, heading_threshold):
    if len(points) <= 2:
        return points
    else:
        max_heading_threshold = 0
        found_index = 0
        for i in range(1, len(points) - 1):
            angle_of_three_points = getAngle(points[i-1], points[i], points[i + 1])
            if angle_of_three_points > max_heading_threshold:
               max_heading_threshold = angle_of_three_points
               found_index = i
        if max_heading_threshold > heading_threshold:
            one = heading_based(points[:found_index], heading_threshold)
            two = heading_based(points[found_index:], heading_threshold)
            one.extend(two)
            return one
        else:
            return [points[0], points[-1]]

if __name__ == "__main__":

    print ("\n")
    print ("                        ------------------------------------------------------------------------------------------------------------")
    print ("                        --------------------------------------- Heading base - Top Down Algorithm ------------------------------------")
    print ("                        ------------------------------------------------------------------------------------------------------------")
    print ("\n")
    currentDirectory = os.getcwd()
    # dynamicPath = os.path.dirname(currentDirectory)

    dynamicPath = currentDirectory + '/Compressed_datasets/HD/'
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
        # the last parameters only used for demonstration reason (initial points vs compressed points)
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
