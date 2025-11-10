# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:54:37 2022

@author: Michael Isselstein
"""


import numpy as np
import pandas as pd


import pims


import trackpy as tp
import os
import time

import matplotlib.pyplot as plt 
import imageio as io
import skimage
from skimage.morphology import disk
from skimage.filters import median
import skimage.measure as measure
import skimage.morphology as morphology



'''
Functions for Image processing
'''
@pims.pipeline
def subtr_backgr(img, backgr):
    return img / backgr

def clean_backgr_img(path, backgr_filename):
    backgr = np.asarray(io.imread(os.path.join(path, backgr_filename)))
    backgr = median(backgr, disk(20))
    backgr = backgr/np.amax(backgr)
    return backgr

@pims.pipeline
def preprocess_bacteria(img, thresh = 530):
    #Convert Image into white with black background
    img = np.amax(img) - img
    #Find local threshold
    thresh = skimage.filters.threshold_local(img, 111)
    #Apply local threshold
    img = img > thresh
    #Use remove_small_object filter to remove falesly detected backgr pixels
    img = skimage.morphology.remove_small_objects(img, 15)
    #Erode image to get rid of thin lines remaining from noise
    img = morphology.binary_erosion(img)
    #Remove remaining objects obtained from noise
    img = skimage.morphology.remove_small_objects(img, 15)
    #Re-dilate the remianing objects
    img = skimage.morphology.binary_dilation(img)
    #Convert from binary image to int-type image
    img = skimage.util.img_as_int(img)
    return img

'''
Save centroids of each detected cell
'''
def cell_detection(path, file, preprocess = "n"):
    print("Starting cell detection")
    start = time.time()
    # features = pd.DataFrame()
    cells_dict = {"x": [], "y": [], "frame": []}
    if preprocess == "n":
        frames = preprocess_bacteria(pims.open(os.path.join(path, file)))
    elif preprocess == "y":
        frames = pims.open(os.path.join(path, file))
    assert preprocess =="y" or preprocess=="n", "Faulty input. Please re-run the cell, choosing y or n as a response"
    for num, img, in enumerate(frames):    
        label_img = measure.label(img)
        for region in measure.regionprops(label_img, intensity_image = img):
            cells_dict["x"].append(region.centroid[1])
            cells_dict["y"].append(region.centroid[0])
            cells_dict["frame"].append(num)
    cells_dict["Movie"] = [file for i in range(len(cells_dict["x"]))]
    cells = pd.DataFrame.from_records(cells_dict)
    print("Finished cell detection. Execution time {:.2f}s".format(time.time() - start))
    return cells

'''
Functions for concentrating data from tracking results
'''
def calc_dx_dy(track):
    track = track[["frame", "x", "y", "particle"]].sort_values("frame").copy()
    delta = (np.roll(track, -1, axis = 0) - track)[:-1]
    dx = delta.x / delta.frame
    dy = delta.y / delta.frame
    return dx, dy

def calc_stepsize(track):
    dx, dy = calc_dx_dy(track)
    steps = (dx**2 + dy**2)**0.5
    return np.mean(steps), np.std(steps)

def calc_scalar_product_angle(track):
    dx, dy = calc_dx_dy(track)
    #calculate scalar product component-wise
    spx = (dx * np.roll(dx, -1))[:-1]
    spy = (dy * np.roll(dy, -1))[:-1]
    #calculate scalar product. Each element of sp represents the scalar product of two supsequent vectors
    sp = spx + spy
    #calculate step-lengths
    vec_len = np.sqrt(dx**2 +dy**2)
    #calculate normalization factor (dx_i**2 +dy_i**2)**0.5 * (dx_i+1**2 + dy_i+1**2)**0.5
    # = vec_len[i] * vec_len[i+1]
    norm = (vec_len * np.roll(vec_len, -1))[:-1]
    angle = np.arccos(sp / norm)
    return angle

def calc_direction(track):
    #calculate directoin from vector product.
    #The sign of the vector product gives the direction of rotation. Negative sign mean cw rotation!
    dx, dy = calc_dx_dy(track)
    #calculate vector product component-wise
    left = (dx * np.roll(dy, -1))[:-1]
    right = (dy * np.roll(dx, -1))[:-1]
    #calculate scalar product. Each element of sp represents the scalar product of two supsequent vectors
    vp = left - right
    #calculate step-lengths
    vec_len = np.sqrt(dx**2 +dy**2)
    #calculate normalization factor (dx_i**2 +dy_i**2)**0.5 * (dx_i+1**2 + dy_i+1**2)**0.5
    # = vec_len[i] * vec_len[i+1]
    norm = (vec_len * np.roll(vec_len, -1))[:-1]
    angle = np.arcsin(vp / norm)
    return np.sign(angle)


def evaluate_tracks(tracks_mult_mov, magnification, px_size, fps, low_angle=20):
    '''

    Parameters
    ----------
    tracks_mult_mov : Pandas Dataframe
        Dataframe containing tracking Data. Must include at least following columns: "x", "y", "Movie", "particle".
    magnification : int or float
        Total magnification of microscope.
    px_size : int or float
        Physical edge-length of camera pixel in the movie.
    fps : int or float
        Frames per second used to record the movie.
    low_angle : int or float, optional
        Threshold angle in degrees - angles below the absolute of low_angle are treated as small angles. The default is 20.

    Returns
    -------
    data : Pandas Dataframe
        Contains all parameters evaluated. See definitoin of "data" variable below.
        
        Movie: Filename of movie for respective particle
        particle: Index of particle
        distance[µm]: Calculated total distance traveled over duration of movie in µm
        track_duration [frames]: duration of track in unit frames
        number_of_frames [frames]: Number of frames that the particle was detected
        track_duration [s]: duration of track in unit seconds
        mean_stepsize[µm]: Average of linear distance between positions in µm
        std_stepsize[µm]: Standard deviation of linear distance between positions in µm
        mean_velocity[µm]: Average of linear distance between positions per second in µm/s
        std_velocity [µm/s]: Standard deviation of linear distance between positions per second in µm/s
        rel_cw_rot: Fraction of steps going in a clockwise direction
        rel_ccw_rot: Fraction of steps going in a counterclockwise direction
        rel_low_dir_ch: Fraction of steps going in angles smaller than defined "low_angle" variable
        av_angle [°]: Average of the angle in Degrees
        std_angle [°]: Standard deviation of the angle in Degrees
        mean_velocity / distance [1/s]: Mean Velocity divided by the total distance. Unit is 1/s
        mean_stepsize / distance: Mean stepsize divided by the total distance. Per definition unitles
        std_position [µm]: Standard Deviation of cell position. Calculated as sqrt(var(x) + var(y)), 
            namely the squareroot of the sum of the positional variances 
    '''
    
    '''
    Convert x and y into µm
    '''
    tracks_mult_mov = tracks_mult_mov.copy()
    tracks_mult_mov.x = tracks_mult_mov.x * px_size / magnification
    tracks_mult_mov.y = tracks_mult_mov.y * px_size / magnification
    '''
    low_angle: angular threshold below which directional change is called "low"
    '''
    low_angle = low_angle / 180 * np.pi
    
    data = {"Movie": [], "particle":[], "distance [µm]":[], "track_duration [frames]":[], "number_of_frames": [], "track_duration [s]":[], "mean_stepsize [µm]": [], "std_stepsize [µm]": [], "mean_velocity [µm/s]": [],\
            "std_velocity [µm/s]": [], "rel_cw_rot": [], "rel_ccw_rot": [], "rel_low_dir_ch": [], "av_angle [°]": [], "std_angle [°]": [], "mean_velocity / distance [1/s]": [], \
            "mean_stepsize / distance":[], "std_position [µm]": []}
    for movie in set(tracks_mult_mov.Movie):
        tracks = tracks_mult_mov[tracks_mult_mov.Movie == movie].copy()
        for part in set(tracks.particle):
            track = tracks[tracks.particle==part].copy().sort_values("frame")
            data["Movie"].append(movie)
            data["particle"].append(part)
            length = np.amax(track.frame) - np.amin(track.frame) + 1
            data["track_duration [frames]"].append(length)
            data["number_of_frames"].append(len(track))
            data["track_duration [s]"].append(length / fps)
            x = track.x.to_numpy()
            y = track.y.to_numpy()
            data["std_position [µm]"].append((np.var(x) + np.var(y))**0.5)
            distance = ((x[0]-x[-1])**2 + (y[0]-y[-1])**2)**0.5
            data["distance [µm]"].append(distance)
            res = calc_stepsize(track)
            data["mean_stepsize [µm]"].append(res[0])
            data["std_stepsize [µm]"].append(res[1])
            data["mean_velocity [µm/s]"].append(res[0] * fps)
            data["std_velocity [µm/s]"].append(res[1] * fps)
            data["mean_velocity / distance [1/s]"].append(res[0] * fps / distance)
            data["mean_stepsize / distance"].append(res[0] / distance)
            angles = calc_scalar_product_angle(track)
            directions = calc_direction(track)
            if len(angles > 0):
                data["rel_cw_rot"].append(np.sum(directions < 0) / len(directions))
                data["rel_ccw_rot"].append(np.sum(directions > 0) / len(directions))
                data["rel_low_dir_ch"].append(np.sum(np.abs(angles) <= low_angle) / len(angles))
                data["av_angle [°]"].append(np.mean(angles * directions)*180/np.pi)
                data["std_angle [°]"].append(np.std(angles * directions)*180/np.pi)
            else:
                data["rel_cw_rot"].append(np.nan)
                data["rel_ccw_rot"].append(np.nan)
                data["rel_low_dir_ch"].append(np.nan)
                data["av_angle [°]"].append(np.nan)
                data["std_angle [°]"].append(np.nan)
        
    data = pd.DataFrame(data)
    return data