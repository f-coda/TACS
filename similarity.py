import numpy as np
import pickle
import sys
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import math
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import os
import geohash2.geohash as geoh


# colors for terminal
CRED = '\033[91m'
CREDEND = '\033[0m'

R = 6378137.0

# load all elements from dataset
def load_data_from_file(provided_dataset):
    # load the point data
    df = pd.read_csv(provided_dataset)

    df['LONGITUDE-LATITUDE'] = list(
        map(list, zip(df['LONGITUDE'], df['LATITUDE'])))
    grouped = df.groupby('mmsi')['LONGITUDE-LATITUDE'].apply(list)
    res_ufc = [[k, v] for k, v in grouped.items()]
    return res_ufc

def load_data_from_file_comp(provided_dataset):
    # load the point data
    df = pd.read_csv(provided_dataset)

    df['longitude-latitude'] = list(
        map(list, zip(df['longitude'], df['latitude'])))
    grouped = df.groupby('mmsi')['longitude-latitude'].apply(list)
    res_ufc = [[k, v] for k, v in grouped.items()]
    return res_ufc
#--------------------------------- distances --------------------------------
def haversine_distance(x, y):
    EARTH_RADIUS = 6371 * 1000

    """Haversine distance between two points, expressed in meters."""
    lat1 = radians(x[1])
    lon1 = radians(x[0])
    lat2 = radians(y[1])
    lon2 = radians(y[0])

    d_lon = lon2 - lon1
    d_lat = lat2 - lat1

    a = math.sin(d_lat/2) * math.sin(d_lat/2) + \
        math.sin(d_lon/2) * math.sin(d_lon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = EARTH_RADIUS * c
    return d

def eucl_dist(x, y):
    dist = np.linalg.norm(x - y)
    return dist


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lon1,lat1) and (lon2,lat2)
    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the se*cond point
    param lon2: float, longitude of the second point
    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """
    rad = math.pi / 180.0

    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def initial_bearing(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Bearing between (lon1,lat1) and (lon2,lat2), in degree.
    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the second point
    param lon2: float, longitude of the second point
    Returns
    -------
    brng: float
           Bearing between (lon1,lat1) and (lon2,lat2), in degree.
    """
    rad = math.pi / 180.0
    dlon = rad * (lon2 - lon1)
    y = math.sin(dlon) * math.cos(rad * (lat2))
    x = math.cos(rad * (lat1)) * math.sin(rad * (lat2)) - math.sin(rad * (lat1)) * math.cos(rad * (lat2)) * math.cos(
        dlon)
    ibrng = math.atan2(y, x)

    return ibrng

def great_circle_distance_traj(lons1, lats1, lons2, lats2, l1, l2):
    """
    Usage
    -----
    Compute pairwise great circle distance, in meter, between longitude/latitudes coordinates.
    Parameters
    ----------
    param lats1: float, latitudes of the firs trajectories
    param lons1: float, longitude of the trajectories
    param lats2: float, latitudes of the se*cond trajectories
    param lons2: float, longitudess of the second trajectories
    param l1 : int, length of the first trajectories
    param l2 : int, length of the second trajectories
    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """
    mdist = np.empty((l1, l2), dtype=float)
    for i in range(l1):
        for j in range(l2):
            mdist[i, j] = great_circle_distance(lons1[i], lats1[i], lons2[j], lats2[j])
    return mdist

def cross_track_distance(lon1, lat1, lon2, lat2, lon3, lat3, d13):
    """
    Usage
    -----
    Angular cross-track-distance of a point (lon3, lat3) from a great-circle path between (lon1, lat1) and (lon2, lat2)
    The sign of this distance tells which side of the path the third point is on.
    Parameters :
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the second point
    param lon2: float, longitude of the second point
     param lat3: float, latitude of the third point
    param lon3: float, longitude of the third point
    Usage
    -----
    crt: float
         the (angular) cross_track_distance
    """

    theta13 = initial_bearing(lon1, lat1, lon3, lat3)  # bearing from start point to third point
    theta12 = initial_bearing(lon1, lat1, lon2, lat2)  # bearing from start point to end point

    crt = math.asin(math.sin(d13 / R) * math.sin(theta13 - theta12)) * R

    return crt


def along_track_distance(crt, d13):
    """
    Usage
    -----
    The along-track distance from the start point (lon1, lat1) to the closest point on the the path
    to the third point (lon3, lat3).
    Parameters
    ----------
    param crt : float, cross_track_distance
    param d13 : float, along_track_distance
    Returns
    -------
    alt: float
         The along-track distance
    """

    alt = math.acos(math.cos(d13 / R) / math.cos(crt / R)) * R
    return alt

def eucl_dist_traj(t1, t2):
    """
    Usage
    -----
    Pairwise L2-norm between point of trajectories t1 and t2
    Parameters
    ----------
    param t1 : len(t1)x2 numpy_array
    param t2 : len(t1)x2 numpy_array
    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    mdist = cdist(t1, t2, 'euclidean')
    return mdist

def circle_line_intersection(px, py, s1x, s1y, s2x, s2y, eps):
    """
    Usage
    -----
    Find the intersections between the circle of radius eps and center (px, py) and the line delimited by points
    (s1x, s1y) and (s2x, s2y).
    It is supposed here that the intersection between them exists. If no, raise error
    Parameters
    ----------
    param px : float, centre's abscissa of the circle
    param py : float, centre's ordinate of the circle
    param eps : float, radius of the circle
    param s1x : abscissa of the first point of the line
    param s1y : ordinate of the first point of the line
    param s2x : abscissa of the second point of the line
    param s2y : ordinate of the second point of the line
    Returns
    -------
    intersect : 2x2 numpy_array
                Coordinate of the two intersections.
                If the two intersections are the same, that's means that the line is a tangent of the circle.
    """
    if s2x == s1x:
        rac = math.sqrt((eps * eps) - ((s1x - px) * (s1x - px)))
        y1 = py + rac
        y2 = py - rac
        intersect = np.array([[s1x, y1], [s1x, y2]])
    else:
        m = (s2y - s1y) / (s2x - s1x)
        c = s2y - m * s2x
        A = m * m + 1
        B = 2 * (m * c - m * py - px)
        C = py * py - eps * eps + px * px - 2 * c * py + c * c
        delta = B * B - 4 * A * C
        if delta <= 0:
            x = -B / (2 * A)
            y = m * x + c
            intersect = np.array([[x, y], [x, y]])
        elif delta > 0:
            sdelta = math.sqrt(delta)
            x1 = (-B + sdelta) / (2 * A)
            y1 = m * x1 + c
            x2 = (-B - sdelta) / (2 * A)
            y2 = m * x2 + c
            intersect = np.array([[x1, y1], [x2, y2]])
        else:
            raise ValueError("The intersection between circle and line is supposed to exist")
    return intersect

def point_to_seg(p, s1, s2, dps1, dps2, ds):
    """
    Usage
    -----
    Point to segment distance between point p and segment delimited by s1 and s2
    Parameters
    ----------
    param p : 1x2 numpy_array
    param s1 : 1x2 numpy_array
    param s2 : 1x2 numpy_array
    dps1 : euclidean distance between p and s1
    dps2 : euclidean distance between p and s2
    dps : euclidean distance between s1 and s2
    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """
    px = p[0]
    py = p[1]
    p1x = s1[0]
    p1y = s1[1]
    p2x = s2[0]
    p2y = s2[1]
    if p1x == p2x and p1y == p2y:
        dpl = dps1
    else:
        segl = ds
        x_diff = p2x - p1x
        y_diff = p2y - p1y
        u1 = (((px - p1x) * x_diff) + ((py - p1y) * y_diff))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            # closest point does not fall within the line segment, take the shorter distance to an endpoint
            dpl = min(dps1, dps2)
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * x_diff
            iy = p1y + u * y_diff
            dpl = eucl_dist(p, np.array([ix, iy]))

    return dpl
#------------------------------------------------------------------------------


########################################################## trajminer ###################################################
########################################################################################################################
'''def edr_new(t1,t2):
    matrix = np.zeros(shape=[len(t1) + 1, len(t2) + 1])
    matrix[:, 0] = np.r_[0:len(t1)+1]
    matrix[0] = np.r_[0:len(t2)+1]

    for i, p1 in enumerate(t1):
        for j, p2 in enumerate(t2):
            cost = _edr_match_cost(p1, p2)
            matrix[i+1][j+1] = min(matrix[i][j] + cost,
                                   min(matrix[i+1][j] + 1,
                                       matrix[i][j+1] + 1))

    print (1 - matrix[len(t1)][len(t2)] / max(len(t1), len(t2)))


def _edr_match_cost(p1, p2):

    for i, _ in enumerate(p1):
        #d = self.dist_functions[i](p1[i], p2[i])
        #if d > self.thresholds[i]:
        if eucl_dist(p1[i], p2[i]) > 0.5:
            break
    else:
        return 0
    return 1


def lcss_new(t1, t2):
    matrix = np.zeros(shape=[2, len(t2) + 1])

    for i, p1 in enumerate(t1):
        ndx = i & 1
        ndx1 = int(not ndx)
        for j, p2 in enumerate(t2):
            if _lcss_match(p1, p2):
                matrix[ndx1][j+1] = matrix[ndx][j] + 1
            else:
                matrix[ndx1][j+1] = max(matrix[ndx1][j], matrix[ndx][j+1])

    print( matrix[1][len(t2)] / min(len(t1), len(t2)))

def _lcss_match(p1, p2):
    for i, _ in enumerate(p1):
        if eucl_dist(p1[i], p2[i]) > 0.5:
            break
    else:
        return True
    return False'''
########################################################################################################################
########################################################################################################################

#----------------------------------- trajectory similarity algorithms --------------------------------------------------

# EDR------------------------
def edr(t0,t1,eps):
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]

    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if haversine_distance(t0[i - 1], t1[j - 1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)

    #print ("1edr-",float(C[n0][n1]))
    #print ("2edr-", max([n0, n1]))
    edr = float(C[n0][n1]) / max([n0, n1])
    #print("EDR", edr)
    return edr
#----------------------------

# LCSS ----------------------
def check_arrays(X, Y):
    """Set X and Y appropriately.

    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :returns: X and Y in 2D numpy arrays
    """
    X = np.array(X, dtype=np.float)
    Y = np.array(Y, dtype=np.float)
    if X.ndim == 1:
        X = np.reshape(X, (1, X.size))
    if Y.ndim == 1:
        Y = np.reshape(Y, (1, Y.size))
    return X, Y

def standardization(X):
    """Transform X to have zero mean and one standard deviation."""
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


def _lcss_dist(X, Y, delta, epsilon):
    """Compute the LCSS distance between X and Y using Dynamic Programming.
    :param X (2-D array): time series feature array denoted by X
    :param Y (2-D array): time series feature array denoted by Y
    :param delta (int): time sample matching threshold
    :param epsilon (float): amplitude matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: M Vlachos et al., "Discovering Similar Multidimensional Trajectories", 2002.
    """
    n_frame_X, n_frame_Y = X.shape[1], Y.shape[1]
    S = np.zeros((n_frame_X+1, n_frame_Y+1))
    for i in range(1, n_frame_X+1):
        for j in range(1, n_frame_Y+1):
            if np.all(np.abs(X[:, i-1]-Y[:, j-1]) < epsilon) and (
                np.abs(i-j) < delta):
                S[i, j] = S[i-1, j-1]+1
            else:
                S[i, j] = max(S[i, j-1], S[i-1, j])
    return 1-S[n_frame_X, n_frame_Y]/min(n_frame_X, n_frame_Y)

def lcss_dist(X, Y, delta, epsilon):
    """Compute the LCSS distance between X and Y using Dynamic Programming.
    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :param delta (int): time sample matching threshold
    :param epsilon (float): amplitude matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: M Vlachos et al., "Discovering Similar Multidimensional Trajectories", 2002
    """
    X, Y = check_arrays(X, Y)
    dist = _lcss_dist(X, Y, delta, epsilon)
    #print ("LCSS", dist)
    return dist
#-----------------------------

# DTW------------------------
def dtw(t0, t1):
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = haversine_distance(t0[i - 1], t1[j - 1]) + min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
    dtw = C[n0, n1]
    #print("DTW", dtw)
    return(dtw)
#---------------------------


# ERP-----------------------
def erp(t0, t1, g):
    dim_1 = t0.shape[1]
    dim_2 = t1.shape[1]

    if dim_1 != 2 or dim_2 != 2:
        raise ValueError("Trajectories should be in 2D. t1 is %dD and t2 is %d given" % (dim_1, dim_2))
    dim = dim_1

    if g is None:
        g = np.zeros(dim, dtype=float)
    else:
        if g.shape[0] != dim:
            raise ValueError("g and trajectories in list should have same dimension")

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))

    C[1:, 0] = sum([abs(great_circle_distance(g[0], g[1], x[0], x[1])) for x in t0])
    C[0, 1:] = sum([abs(great_circle_distance(g[0], g[1], y[0], y[1])) for y in t1])
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            derp0 = C[i - 1, j] + great_circle_distance(t0[i - 1][0], t0[i - 1][1], g[0], g[1])
            derp1 = C[i, j - 1] + great_circle_distance(g[0], g[1], t1[j - 1][0], t1[j - 1][1])
            derp01 = C[i - 1, j - 1] + great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1])
            C[i, j] = min(derp0, derp1, derp01)
    erp = C[n0, n1]
    #print ("ERP", erp)
    return erp
#---------------------------

# frechet-----------------------
def free_line(p, eps, s, dps1, dps2, ds):
    """
    Usage
    -----
    Return the free space in the segment s, from point p.
    This free space is the set of all point in s whose distance from p is at most eps.
    Since s is a segment, the free space is also a segment.
    We return a 1x2 array whit the fraction of the segment s which are in the free space.
    If no part of s are in the free space, return [-1,-1]
    Parameters
    ----------
    param p : 1x2 numpy_array, centre of the circle
    param eps : float, radius of the circle
    param s : 2x2 numpy_array, line
    Returns
    -------
    lf : 1x2 numpy_array
         fraction of segment which is in the free space (i.e [0.3,0.7], [0.45,1], ...)
         If no part of s are in the free space, return [-1,-1]
    """
    px = p[0]
    py = p[1]
    s1x = s[0, 0]
    s1y = s[0, 1]
    s2x = s[1, 0]
    s2y = s[1, 1]
    if s1x == s2x and s1y == s2y:
        if eucl_dist(p, s[0]) > eps:
            lf = [-1, -1]
        else:
            lf = [0, 1]
    else:
        if point_to_seg(p, s[0], s[1], dps1, dps2, ds) > eps:
            # print("No Intersection")
            lf = [-1, -1]
        else:
            segl = eucl_dist(s[0], s[1])
            segl2 = segl * segl
            intersect = circle_line_intersection(px, py, s1x, s1y, s2x, s2y, eps)
            if intersect[0][0] != intersect[1][0] or intersect[0][1] != intersect[1][1]:
                i1x = intersect[0, 0]
                i1y = intersect[0, 1]
                u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y))) / segl2

                i2x = intersect[1, 0]
                i2y = intersect[1, 1]
                u2 = (((i2x - s1x) * (s2x - s1x)) + ((i2y - s1y) * (s2y - s1y))) / segl2
                ordered_point = sorted((0, 1, u1, u2))
                lf = ordered_point[1:3]
            else:
                if px == s1x and py == s1y:
                    lf = [0, 0]
                elif px == s2x and py == s2y:
                    lf = [1, 1]
                else:
                    i1x = intersect[0][0]
                    i1y = intersect[0][1]
                    u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y))) / segl2
                    if 0 <= u1 <= 1:
                        lf = [u1, u1]
                    else:
                        lf = [-1, -1]
    return lf


def LF_BF(P, Q, p, q, eps, mdist, P_dist, Q_dist):
    """
    Usage
    -----
    Compute all the free space on the boundary of cells in the diagram for polygonal chains P and Q and the given eps
    LF[(i,j)] is the free space of segment [Pi,Pi+1] from point  Qj
    BF[(i,j)] is the free space of segment [Qj,Qj+1] from point Pj
    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance
    mdist : p x q numpy array, pairwise distance between points of trajectories t1 and t2
    param P_dist:  p x 1 numpy_array,  distances between consecutive points in P
    param Q_dist:  q x 1 numpy_array,  distances between consecutive points in Q
    Returns
    -------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    """
    LF = {}
    for j in range(q):
        for i in range(p - 1):
            LF.update({(i, j): free_line(Q[j], eps, P[i:i + 2], mdist[i, j], mdist[i + 1, j], P_dist[i])})
    BF = {}
    for j in range(q - 1):
        for i in range(p):
            BF.update({(i, j): free_line(P[i], eps, Q[j:j + 2], mdist[i, j], mdist[i, j + 1], Q_dist[j])})
    return LF, BF


def LR_BR(LF, BF, p, q):
    """
    Usage
    -----
    Compute all the free space,that are reachable from the origin (P[0,0],Q[0,0]) on the boundary of cells
    in the diagram for polygonal chains P and Q and the given free spaces LR and BR
    LR[(i,j)] is the free space, reachable from the origin, of segment [Pi,Pi+1] from point  Qj
    BR[(i,j)] is the free space, reachable from the origin, of segment [Qj,Qj+1] from point Pj
    Parameters
    ----------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    LR : dict, is the free space, reachable from the origin, of segments of P from points of Q
    BR : dict, is the free space, reachable from the origin, of segments of Q from points of P
    """
    if not (LF[(0, 0)][0] <= 0 and BF[(0, 0)][0] <= 0 and LF[(p - 2, q - 1)][1] >= 1 and BF[(p - 1, q - 2)][1] >= 1):
        rep = False
        BR = {}
        LR = {}
    else:
        LR = {(0, 0): True}
        BR = {(0, 0): True}
        for i in range(1, p - 1):
            if LF[(i, 0)] != [-1, -1] and LF[(i - 1, 0)] == [0, 1]:
                LR[(i, 0)] = True
            else:
                LR[(i, 0)] = False
        for j in range(1, q - 1):
            if BF[(0, j)] != [-1, -1] and BF[(0, j - 1)] == [0, 1]:
                BR[(0, j)] = True
            else:
                BR[(0, j)] = False
        for i in range(p - 1):
            for j in range(q - 1):
                if LR[(i, j)] or BR[(i, j)]:
                    if LF[(i, j + 1)] != [-1, -1]:
                        LR[(i, j + 1)] = True
                    else:
                        LR[(i, j + 1)] = False
                    if BF[(i + 1, j)] != [-1, -1]:
                        BR[(i + 1, j)] = True
                    else:
                        BR[(i + 1, j)] = False
                else:
                    LR[(i, j + 1)] = False
                    BR[(i + 1, j)] = False
        rep = BR[(p - 2, q - 2)] or LR[(p - 2, q - 2)]
    return rep, LR, BR


def decision_problem(P, Q, p, q, eps, mdist, P_dist, Q_dist):
    """
    Usage
    -----
    Test is the frechet distance between trajectories P and Q are inferior to eps
    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance
    mdist : p x q numpy array, pairwise distance between points of trajectories t1 and t2
    param P_dist:  p x 1 numpy_array,  distances between consecutive points in P
    param Q_dist:  q x 1 numpy_array,  distances between consecutive points in Q
    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    """
    LF, BF = LF_BF(P, Q, p, q, eps, mdist, P_dist, Q_dist)
    rep, _, _ = LR_BR(LF, BF, p, q)
    return rep


def compute_critical_values(P, Q, p, q, mdist, P_dist, Q_dist):
    """
    Usage
    -----
    Compute all the critical values between trajectories P and Q
    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : int, number of points in Trajectory P
    param q : int, number of points in Trajectory Q
    mdist : p x q numpy array, pairwise distance between points of trajectories t1 and t2
    param P_dist:  p x 1 numpy_array,  distances between consecutive points in P
    param Q_dist:  q x 1 numpy_array,  distances between consecutive points in Q
    Returns
    -------
    cc : list, all critical values between trajectories P and Q
    """
    origin = eucl_dist(P[0], Q[0])
    end = eucl_dist(P[-1], Q[-1])
    end_point = max(origin, end)
    cc = set([end_point])
    for i in range(p - 1):
        for j in range(q - 1):
            Lij = point_to_seg(Q[j], P[i], P[i + 1], mdist[i, j], mdist[i + 1, j], P_dist[i])
            if Lij > end_point:
                cc.add(Lij)
            Bij = point_to_seg(P[i], Q[j], Q[j + 1], mdist[i, j], mdist[i, j + 1], Q_dist[j])
            if Bij > end_point:
                cc.add(Bij)
    return sorted(list(cc))


def e_dtw(t0, t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = eucl_dist(t0[i - 1], t1[j - 1]) + min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
    dtw = C[n0, n1]
    return dtw


def discret_frechet_eu(t0, t1):
    """
    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q
    Parameters
    ----------
    param t0 : px2 numpy_array, Trajectory t0
    param t1 : qx2 numpy_array, Trajectory t1
    Returns
    -------
    frech : float, the discret frechet distance between trajectories t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = max(eucl_dist(t0[i - 1], t1[j - 1]), min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j]))
    dtw = C[n0, n1]
    return dtw


def frechet(P, Q):
    """
    Usage
    -----
    Compute the frechet distance between trajectories P and Q
    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    Returns
    -------
    frech : float, the frechet distance between trajectories P and Q
    """
    p = len(P)
    q = len(Q)

    mdist = eucl_dist_traj(P, Q)
    P_dist = [eucl_dist(P[ip], P[ip + 1]) for ip in range(p - 1)]
    Q_dist = [eucl_dist(Q[iq], Q[iq + 1]) for iq in range(q - 1)]

    cc = compute_critical_values(P, Q, p, q, mdist, P_dist, Q_dist)
    eps = cc[0]
    while (len(cc) != 1):
        m_i = len(cc) / 2 - 1
        eps = cc[m_i]
        rep = decision_problem(P, Q, p, q, eps, mdist, P_dist, Q_dist)
        if rep:
            cc = cc[:m_i + 1]
        else:
            cc = cc[m_i + 1:]
    frech = eps
    #print ("frechet", frech)
    return frech
#---------------------------

# hausdorff-----------------------
def point_to_path(lon1, lat1, lon2, lat2, lon3, lat3, d13, d23, d12):
    """
    Usage
    -----
    The point-to-path distance between point (lon3, lat3) and path delimited by (lon1, lat1) and (lon2, lat2).
    The point-to-path distance is the cross_track distance between the great circle path if the projection of
    the third point lies on the path. If it is not on the path, return the minimum of the
    great_circle_distance between the first and the third or the second and the third point.
    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the second point
    param lon2: float, longitude of the second point
    param lat3: float, latitude of the third point
    param lon3: float, longitude of the third point
    param d13 : float, great circle distance between (lon1, lat1) and (lon3, lat3)
    param d23 : float, great circle distance between (lon2, lat2) and (lon3, lat3)
    param d12 : float, great circle distance between (lon1, lat1) and (lon2, lat2)
    Returns
    -------
    ptp : float
          The point-to-path distance between point (lon3, lat3) and path delimited by (lon1, lat1) and (lon2, lat2)
    """
    crt = cross_track_distance(lon1, lat1, lon2, lat2, lon3, lat3, d13)
    d1p = along_track_distance(crt, d13)
    d2p = along_track_distance(crt, d23)
    if (d1p > d12) or (d2p > d12):
        ptp = np.min((d13, d23))
    else:
        ptp = np.abs(crt)
    return ptp

def s_directed_hausdorff(lons0, lats0, lons1, lats1, n0, n1, mdist, t0_dist):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.
    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array
    Returns
    -------
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """

    dh = 0
    for j in range(n1):
        dist_j0 = 9e100
        for i in range(n0 - 1):
            dist_j0 = min(dist_j0, point_to_path(lons0[i], lats0[i], lons0[i + 1], lats0[i + 1], lons1[j],
                                                 lats1[j], mdist[i, j], mdist[i + 1, j], t0_dist[i]))
        dh = max(dh, dist_j0)
    return dh


def hausdorff(t0, t1):

    n0 = len(t0)
    n1 = len(t1)
    lats0 = t0[:, 1]
    lons0 = t0[:, 0]
    lats1 = t1[:, 1]
    lons1 = t1[:, 0]

    mdist = great_circle_distance_traj(lons0, lats0, lons1, lats1, n0, n1)

    t0_dist = [great_circle_distance(lons0[it0], lats0[it0], lons0[it0 + 1], lats0[it0 + 1]) for it0 in range(n0 - 1)]
    t1_dist = [great_circle_distance(lons1[it1], lats1[it1], lons1[it1 + 1], lats1[it1 + 1]) for it1 in range(n1 - 1)]

    h = max(s_directed_hausdorff(lons0, lats0, lons1, lats1, n0, n1, mdist, t0_dist),
            s_directed_hausdorff(lons1, lats1, lons0, lats0, n1, n0, mdist.T, t1_dist))
    #print ("hausdorff", h)
    return h


#-------------------------------
def s_spd(lons0, lats0, lons1, lats1, n0, n1, mdist, t0_dist):
    """
    Usage
    -----
    The spd-distance of trajectory t1 from trajectory t0
    The spd-distance is the sum of the all the point-to-path distance of points of t0 from trajectory t1
    Parameters
    ----------
    param lons0 :  n0 x 1 numpy_array, longitudes of trajectories t0
    param lats0 :  n0 x 1 numpy_array, lattitudes of trajectories t0
    param lons1 :  n1 x 1 numpy_array, longitudes of trajectories t1
    param lats1 :  n1 x 1 numpy_array, lattitudes of trajectories t1
    param n0: int, length of lons0 and lats0
    param n1: int, length of lons1 and lats1
    mdist : len(t0) x len(t1) numpy array, pairwise distance between points of trajectories t0 and t1
    param t0_dist:  l_t1 x 1 numpy_array,  distances between consecutive points in t0
    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """

    dist = 0
    for j in range(n1):
        dist_j0 = 9e100
        for i in range(n0 - 1):
            dist_j0 = np.min((dist_j0, point_to_path(lons0[i], lats0[i], lons0[i + 1], lats0[i + 1], lons1[j],
                                                     lats1[j], mdist[i, j], mdist[i + 1, j], t0_dist[i])))
        dist = dist + dist_j0
    dist = float(dist) / n1
    return dist


def s_sspd(t0, t1):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance is the mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.
    Parameters
    ----------
    param t0 :  len(t0)x2 numpy_array
    param t1 :  len(t1)x2 numpy_array
    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """
    n0 = len(t0)
    n1 = len(t1)
    lats0 = t0[:, 1]
    lons0 = t0[:, 0]
    lats1 = t1[:, 1]
    lons1 = t1[:, 0]

    mdist = great_circle_distance_traj(lons0, lats0, lons1, lats1, n0, n1)

    t0_dist = [great_circle_distance(lons0[it0], lats0[it0], lons0[it0 + 1], lats0[it0 + 1]) for it0 in range(n0 - 1)]
    t1_dist = [great_circle_distance(lons1[it1], lats1[it1], lons1[it1 + 1], lats1[it1 + 1]) for it1 in range(n1 - 1)]

    dist = s_spd(lons0, lats0, lons1, lats1, n0, n1, mdist, t0_dist) + s_spd(lons1, lats1, lons0, lats0, n1, n0,
                                                                             mdist.T, t1_dist)
    return dist


def find_percentiles (list_of_normalized_similarities):
    ten    = np.percentile(list_of_normalized_similarities, 10)
    twenty = np.percentile(list_of_normalized_similarities, 20)
    thirty = np.percentile(list_of_normalized_similarities, 30)
    fourty = np.percentile(list_of_normalized_similarities, 40)
    fifty  = np.percentile(list_of_normalized_similarities, 50)
    sixty  = np.percentile(list_of_normalized_similarities, 60)
    seventy= np.percentile(list_of_normalized_similarities, 70)
    eighty = np.percentile(list_of_normalized_similarities, 80)
    ninty  = np.percentile(list_of_normalized_similarities, 90)

    return ten,twenty,thirty,fourty,fifty,sixty,seventy,eighty,ninty

#----------------------------------------------- main ------------------------------------------------------------------
#=======================================================================================================================
def traj_similarity():

    fetch_ids_and_pairs_from_uncompressed = load_data_from_file(sys.argv[1])
    fetch_ids_and_pairs_from_compressed = load_data_from_file_comp(sys.argv[2])

    coord_list_uncompressed = []
    for i in range(len(fetch_ids_and_pairs_from_uncompressed)):
        coord_list_uncompressed.append(fetch_ids_and_pairs_from_uncompressed[i][1])

    coord_list_compressed = []
    for i in range(len(fetch_ids_and_pairs_from_compressed)):
        coord_list_compressed.append(fetch_ids_and_pairs_from_compressed[i][1])

    #edr_res = []
    dtw_res = []
    #lcss_res =[]
    erp_res = []
    #frechet_res = []
    discret_frechet = []
    hausdorff_res = []
    s_sspd_res = []


    for i in range(len(fetch_ids_and_pairs_from_uncompressed)):
        #edr_res.append(edr(np.asarray(coord_list_uncompressed[i]), np.asarray(coord_list_compressed[i]),0.5))
        #lcss_res.append(lcss_dist([item for sublist in coord_list_compressed[i] for item in sublist],([item for sublist in coord_list_uncompressed[i] for item in sublist]), delta=np.inf, epsilon=0.5))
        dtw_res.append(dtw(np.asarray(coord_list_uncompressed[i]), np.asarray(coord_list_compressed[i])))
        erp_res.append(erp(np.asarray(coord_list_uncompressed[i]), np.asarray(coord_list_compressed[i]),None))
        #frechet_res.append(frechet(np.asarray(coord_list_uncompressed[i]), np.asarray(coord_list_compressed[i])))
        discret_frechet.append(discret_frechet_eu(np.asarray(coord_list_uncompressed[i]), np.asarray(coord_list_compressed[i])))
        hausdorff_res.append(hausdorff(np.asarray(coord_list_uncompressed[i]), np.asarray(coord_list_compressed[i])))
        s_sspd_res.append(s_sspd(np.asarray(coord_list_uncompressed[i]), np.asarray(coord_list_compressed[i])))


    print ("\n")
    print(CRED + "                  Trajectory Similarity results" + CREDEND)
    print ("\n")

    '''normalized_edr = preprocessing.normalize([edr_res])
    print(CRED + "EDR results"+ CREDEND)
    for i in range(len(normalized_edr[0])):
        print("{:f}".format(float(normalized_edr[0][i])))
    print("mean edr: ", sum(normalized_edr[0]) / len(normalized_edr[0]))
    print ("std edr: ", np.std(normalized_edr))
    print("------\n")

    normalized_lcss = preprocessing.normalize([lcss_res])
    print(CRED + "LCSS results"+ CREDEND)
    for i in range(len(normalized_lcss[0])):
        print("{:f}".format(float(normalized_lcss[0][i])))
    print("mean lcss: ", sum(normalized_lcss[0]) / len(normalized_lcss[0]))
    print("std lcss: ", np.std(normalized_lcss))
    print("------\n")'''


    normalized_dtw = preprocessing.normalize([dtw_res])
    print (CRED + "DTW results"+ CREDEND)
    for i in range(len(normalized_dtw[0])):
        print("{:f}".format(float(normalized_dtw[0][i])))
    print("mean dtw: ", sum(normalized_dtw[0]) / len(normalized_dtw[0]))
    print("std dtw: ", np.std(normalized_dtw))
    print ("------\n")

    normalized_erp = preprocessing.normalize([erp_res])
    print(CRED + "ERP results"+ CREDEND)
    for i in range(len(normalized_erp[0])):
        print("{:f}".format(float(normalized_erp[0][i])))
    print("mean erp: ", sum(normalized_erp[0]) / len(normalized_erp[0]))
    print("std erp: ", np.std(normalized_erp))
    print("------\n")

    # normalized_frechet = preprocessing.normalize([frechet_res])
    # print(CRED + "Frechet results"+ CREDEND)
    # for i in range(len(normalized_frechet[0])):
    #     print("{:f}".format(float(normalized_frechet[0][i])))
    # print("mean frechet: ", sum(normalized_frechet[0]) / len(normalized_frechet[0]))
    # print("std frechet: ", np.std(normalized_frechet))
    # print("------\n")

    normalized_disc_frechet = preprocessing.normalize([discret_frechet])
    print(CRED + "Discret Frechet results" + CREDEND)
    for i in range(len(normalized_disc_frechet[0])):
        print("{:f}".format(float(normalized_disc_frechet[0][i])))
    print("mean discret frechet: ", sum(normalized_disc_frechet[0]) / len(normalized_disc_frechet[0]))
    print("std discret frechet: ", np.std(normalized_disc_frechet))
    print("------\n")

    normalized_hausdorff = preprocessing.normalize([hausdorff_res])
    print(CRED + "Hausdorff results"+ CREDEND)
    for i in range(len(normalized_hausdorff[0])):
        print("{:f}".format(float(normalized_hausdorff[0][i])))
    print("mean hausdorff: ", sum(normalized_hausdorff[0]) / len(normalized_hausdorff[0]))
    print("std hausdorff: ", np.std(normalized_hausdorff))
    print("------\n")

    normalized_sspd = preprocessing.normalize([s_sspd_res])
    print(CRED + "SSPD results" + CREDEND)
    for i in range(len(normalized_sspd[0])):
        print("{:f}".format(float(normalized_sspd[0][i])))
    print("mean sspd: ", sum(normalized_sspd[0]) / len(normalized_sspd[0]))
    print("std sspd: ", np.std(normalized_sspd))
    print("------\n")

    print("===============================================\n")


    currentDirectory = os.getcwd()
    dynamicPath = currentDirectory
    file_for_similarity_metrics = dynamicPath+"/"+"results"+sys.argv[3]+".txt"
    print (file_for_similarity_metrics)
    with open(file_for_similarity_metrics, 'a') as out:
        '''out.write("mean edr: "+ str(sum(normalized_edr[0]) / len(normalized_edr[0])) + '\n')
        out.write("std edr: "+ str(np.std(normalized_edr)) + '\n\n')

        out.write("mean lcss: " + str(sum(normalized_lcss[0]) / len(normalized_lcss[0])) + '\n')
        out.write("std lcss: "+ str(np.std(normalized_lcss))+ '\n\n')'''

        out.write("mean dtw: " + str(sum(normalized_dtw[0]) / len(normalized_dtw[0])) + '\n')
        out.write("std dtw: "+ str(np.std(normalized_dtw)) + '\n')
        out.write("percentiles from 10% to 90%: "+ str(find_percentiles(normalized_dtw)) + '\n\n')

        out.write("mean erp: "+ str(sum(normalized_erp[0]) / len(normalized_erp[0])) + '\n')
        out.write("std erp: "+ str(np.std(normalized_erp)) + '\n')
        out.write("percentiles from 10% to 90%: " + str(find_percentiles(normalized_erp)) + '\n\n')

        # out.write("mean frechet: " + str(sum(normalized_frechet[0]) / len(normalized_frechet[0])) + '\n')
        # out.write("std frechet: "+ str(np.std(normalized_frechet)) + '\n')
        # out.write("percentiles from 10% to 90%: " + str(find_percentiles(normalized_frechet)) + '\n\n')


        out.write("mean discret frechet: " + str(sum(normalized_disc_frechet[0]) / len(normalized_disc_frechet[0])) + '\n')
        out.write("std discret frechet: " + str(np.std(normalized_disc_frechet)) + '\n')
        out.write("percentiles from 10% to 90%: " + str(find_percentiles(normalized_disc_frechet)) + '\n\n')

        out.write("mean hausdorff: " + str(sum(normalized_hausdorff[0]) / len(normalized_hausdorff[0])) + '\n')
        out.write("std hausdorff: "+ str(np.std(normalized_hausdorff)) + '\n')
        out.write("percentiles from 10% to 90%: " + str(find_percentiles(normalized_hausdorff)) + '\n\n')

        out.write("mean sspd: " + str(sum(normalized_sspd[0]) / len(normalized_sspd[0])) + '\n')
        out.write("std sspd: " + str(np.std(normalized_sspd)) + '\n')
        out.write("percentiles from 10% to 90%: " + str(find_percentiles(normalized_sspd)) + '\n\n')

    out.close()

if __name__ == "__main__":
    traj_similarity()
