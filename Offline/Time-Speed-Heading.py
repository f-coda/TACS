import os
import sys
from pointClass import Point
from pointClass import Point_ref
from pointClass import Point_write_to_csv
import pandas as pd
import time
import math

I_3600 = 1 / 3600.0

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

def convert_array_to_pointobj(points, unique_mmsi ,numberofinitialpoints,dynamicPath):
    # split the array
    obj_points = []
    array_points = [c for c in points[0]]

    # convert to Point obj
    for i in range(len(array_points)):
        obj_points.append(
            Point_write_to_csv(array_points[i][0], array_points[i][1], array_points[i][2], array_points[i][3],
                                  array_points[i][4], array_points[i][5]))

    define_thresholds_for_speed_and_heading = define_thresholds_for_time_speed_and_heading_threshold(obj_points)

    #print "-",define_thresholds_for_speed_and_heading[0],"-",define_thresholds_for_speed_and_heading[1],"-",define_thresholds_for_speed_and_heading[2]
    try:

        # Call the time ratio speed function with the data in the correct form (Point obj)
        objects_dynamic_rdp = time_speed_heading(obj_points, define_thresholds_for_speed_and_heading[0],
                                              define_thresholds_for_speed_and_heading[1],define_thresholds_for_speed_and_heading[2])
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
            df.to_csv(dynamicPath + 'Time_Speed_Heading_AVG.csv', mode='a', index=False)
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
            df.to_csv(dynamicPath + 'Time_Speed_Heading_AVG.csv', mode='a', index=False, header=False)

        print(" \n MMSI: ", unique_mmsi, "--Coordinate pairs after compression: ", len(
            objects_dynamic_rdp), " Initial points: ", numberofinitialpoints)

        print("\n", (1 - float(len(objects_dynamic_rdp)) / float(numberofinitialpoints)) * 100, 'percent compressed')
        print("------------------------------------")
        list_with_compressed_points_in_len = gather_compressed_pointsLen_in_each_run(len(objects_dynamic_rdp))
        # return the length of compressed points, the sum
        return sum(list_with_compressed_points_in_len)
    except:
        pass

# Temporal distance between two points (end-start)
# Input: two Points
# Output: time difference in seconds (float)
def time_dist(end, start):
    return end.time_difference(start)

# Spatial distance between two points (end-start)
# Input: Two Points
# Output:  distance in meters (float)
def loc_dist(end, start):
    return end.distance(start)

# Law of Cosines
def getAngle(a, b, c):
    #latitude/longitude
    ang = math.degrees(math.atan2(c.longitude-b.longitude, c.latitude-b.latitude) - math.atan2(a.longitude-b.longitude, a.latitude-b.latitude))
    return ang + 360 if ang < 0 else ang

def define_thresholds_for_time_speed_and_heading_threshold(points):
    list_with_speeds = []

    if len(points) <= 2:
        return points
    else:

        for i in range(1, len(points) - 1):
            dt1 = time_dist(points[i], points[i - 1])
            if dt1 == 0:
                dt1 = 0.000000001
            vim = loc_dist(points[i], points[i - 1]) / dt1
            # print vim
            list_with_speeds.append(vim)

        average_speed = sum(list_with_speeds) / float(len(list_with_speeds))

        list_with_headings = []

        for i in range(1, len(points)-1):
            list_with_headings.append(getAngle( points[i-1],  points[i],  points[i+1]))
        average_heading = sum(list_with_headings) / float(len(list_with_headings))

        list_with_time = []
        # De = te - ts (time difference between end point and start point in seconds)
        delta_e = time_dist(points[-1], points[0]) * I_3600
        # (xe - xs)
        d_lat = points[-1].latitude - points[0].latitude
        # (ye - ys)
        d_lon = points[-1].longitude - points[0].longitude

        for i in range(1, len(points) - 1):
            # Di = ti - ts (time difference between Point examined and start point)
            delta_i = time_dist(points[i], points[0]) * I_3600
            try:
                # Di/De (time interval)
                di_de = delta_i / delta_e
            except ZeroDivisionError:
                # if delta_e = 0 return the points inserted for this id, there is no compression
                return points

            # find the xi' and yi', the approximation point Pi'
            # xi' = xs + Di/De* (xe - xs)
            # yi' = ys + Di/De * (ye - ys)
            point = Point(
                points[0].latitude + d_lat * di_de,
                points[0].longitude + d_lon * di_de,
                None
            )

            # find the distance between it (Pi') and the original Pi in meters
            dist = loc_dist(points[i], point)
            list_with_time.append(dist)

        average_time = sum(list_with_time) / float(len(list_with_time))
        return average_time,average_speed,average_heading

def time_speed_heading(points,max_time_error,max_speed_error,max_heading_error):

    if len(points) <= 2:
        return points
    else:
        is_error = False
        e = 1
        while e < len(points) and not is_error:
            i = 1
            while i < e and not is_error:

                dt1 = time_dist(points[i], points[i-1])
                if dt1 == 0:
                    dt1 = 0.000000001
                dt2 = time_dist(points[i+1], points[i])
                if dt2 == 0:
                    dt2 = 0.000000001

                v_i_1 = loc_dist(points[i], points[i-1]) / dt1
                v_i = loc_dist(points[i+1], points[i]) / dt2

                delta_e = time_dist(points[e], points[0]) * I_3600
                delta_i = time_dist(points[i], points[0]) * I_3600

                di_de = 0
                if delta_e != 0:
                    di_de = delta_i / delta_e
                d_lat = points[e].latitude - points[0].latitude
                d_lon = points[e].longitude - points[0].longitude
                point = Point(
                    points[0].latitude + d_lat * di_de,
                    points[0].longitude + d_lon * di_de,
                    None
                )

                if  loc_dist(points[i], point) > max_time_error or abs(v_i - v_i_1) > max_speed_error or getAngle(points[i - 1], points[i], points[i + 1]) > max_heading_error:
                    is_error = True
                else:
                    i = i + 1
            if is_error:
                return [points[0]] + time_speed_heading(points[i:len(points)],max_time_error, max_speed_error, max_heading_error)
            e = e + 1
        if not is_error:
            return [points[0], points[len(points)-1]]

if __name__ == "__main__":

    print ("\n")
    print ("                        ------------------------------------------------------------------------------------------------------------")
    print ("                        --------------------------------------- Time/Speed/Heading- Top Down Algorithm -----------------------------")
    print ("                        ------------------------------------------------------------------------------------------------------------")
    print ("\n")

    currentDirectory = os.getcwd()
    # dynamicPath = os.path.dirname(currentDirectory)

    dynamicPath = currentDirectory + '/Compressed_datasets/TSH/'
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

    # Call function rdp_dynamic which takes as input a list of Points from dataset
    # Thus at first, the convert_array_to_pointobj function must be called
    # The list data must be converted to Point objects, that's the reason
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
