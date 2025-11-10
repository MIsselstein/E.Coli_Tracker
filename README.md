# E.Coli_Tracker
Python-based tools for detecting and tracking cells in microscopy image sequences as in Bottlinger et.al.



This repository contains:

Cell_tracking.ipynb – Jupyter notebook for applying and visualizing the tracking workflow.

Tracking_functions.py – a collection of functions for background correction, cell segmentation, feature extraction, and trajectory analysis.

The scripts use trackpy, pims, scikit-image, and pandas to extract cell positions from image stacks, link trajectories, and compute motion statistics such as step size, angular change, and rotation direction.
