# coding=utf-8
import math
import datetime
import time
from datetime import datetime

# Spatiotemporal point representation
class Point(object):
    def __init__(self, latitude, longitude, timestamp):
        self.longitude = longitude
        self.latitude = latitude
        self.timestamp = timestamp

    # distance between two points
    # Output: distance in km
    def distance(self, other):
        return distance(self.latitude, self.longitude, None, other.latitude, other.longitude, None)
    # Calculates the time difference in seconds against another point
    def time_difference(self, previous):
        time_previous = datetime.strptime(previous.timestamp, "%Y-%m-%d %H:%M:%S")
        tp = time.mktime(time_previous.timetuple())
        time_n = datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")
        tn = time.mktime(time_n.timetuple())
        return abs((tn - tp))

# Classes in order to covert lists into Point objects for each algorithm
class Point_ref(object):
    def __init__(self, latlonlist):
        self.latlonlist = latlonlist
    def __repr__(self):
        return "<Point latlonlist:%s >" % (self.latlonlist)
    def __str__(self):
        return "Point: latlonlist is %s" % (self.latlonlist)
    def __getitem__(self, key):
        return self.latlonlist[key]
#=======================================================================================================================
#=======================================================================================================================

class Point_write_to_csv(object):
    def __init__(self, latitude, longitude, cog, sog, timestamp, mmsi):
        self.latitude = latitude
        self.longitude = longitude
        self.cog = cog
        self.sog = sog
        self.timestamp = timestamp
        self.mmsi = mmsi
    # distance between two points
    # Output: distance in km
    def distance(self, other):
        return distance(self.latitude, self.longitude, None, other.latitude, other.longitude, None)
    # Calcultes the time difference in seconds against another point
    def time_difference(self, previous):
        time_previous = datetime.strptime(previous.timestamp, "%Y-%m-%d %H:%M:%S")
        tp = time.mktime(time_previous.timetuple())
        time_n = datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")
        tn = time.mktime(time_n.timetuple())
        return abs((tn - tp))
#=======================================================================================================================
#=======================================================================================================================

# Functions to measure the distance between two points
ONE_DEGREE = 1000. * 10000.8 / 90.
EARTH_RADIUS = 6371 * 1000
def to_rad(number):
    """ Degrees to rads """
    return number / 180. * math.pi

def haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2):

    """Haversine distance between two points, expressed in meters."""
    d_lat = to_rad(latitude_1 - latitude_2)
    d_lon = to_rad(longitude_1 - longitude_2)
    lat1 = to_rad(latitude_1)
    lat2 = to_rad(latitude_2)

    a = math.sin(d_lat/2) * math.sin(d_lat/2) + \
        math.sin(d_lon/2) * math.sin(d_lon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = EARTH_RADIUS * c
    return d

def distance(latitude_1, longitude_1, elevation_1, latitude_2, longitude_2, elevation_2,
             haversine=None):
    """ Distance between two points """
    # If points too distant -- compute haversine distance:
    if haversine or (abs(latitude_1 - latitude_2) > .2 or abs(longitude_1 - longitude_2) > .2):
        return haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2)

    coef = math.cos(latitude_1 / 180. * math.pi)
    x = latitude_1 - latitude_2
    y = (longitude_1 - longitude_2) * coef
    distance_2d = math.sqrt(x * x + y * y) * ONE_DEGREE
    if elevation_1 is None or elevation_2 is None or elevation_1 == elevation_2:
        return distance_2d
    return math.sqrt(distance_2d ** 2 + (elevation_1 - elevation_2) ** 2)