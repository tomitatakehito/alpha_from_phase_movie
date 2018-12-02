import os,sys
from os import path,walk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
from scipy.interpolate import UnivariateSpline as spline
import math

#read phase movie
def read_movie(wdir,movie_name):
	phase_movie_path = os.path.join( wdir, movie_name )
	phase_movie = io.imread(phase_movie_path, plugin="tifffile")

	return phase_movie

#create vectors
def create_vector_dictionary(phase_snap,delta):	#phase_snap is a single frame of a phase_movie
	print('extracting vectors...')
	ydim,xdim = phase_snap.shape
	x_vec_dic = {}
	x_vec_positions = range(0,ydim,delta)
	for k in x_vec_positions:
		x_vec = np.unwrap(phase_snap[k,:])
		x_vec_dic[k] = x_vec

	y_vec_dic = {}
	y_vec_positions = range(0,xdim,delta)
	for k in y_vec_positions:
		y_vec = np.unwrap(phase_snap[:,k])
		y_vec_dic[k] = y_vec

	return x_vec_dic,y_vec_dic

#create splines
def create_spline_dictionary(x_vec_dic,y_vec_dic,smoothing=0.5,k_value=3):
	print('creating splines...')
	x_vec_spline_dic = {}
	space = np.arange(0,len(x_vec_dic[0]),1)
	for key in x_vec_dic:
		x_vec_spline = spline(space,x_vec_dic[key], s = smoothing, k = k_value)
		x_vec_spline_dic[key] = x_vec_spline

	y_vec_spline_dic = {}
	space = np.arange(0,len(y_vec_dic[0]),1)
	for key in y_vec_dic:
		y_vec_spline = spline(space,y_vec_dic[key], s = smoothing, k = k_value)
		y_vec_spline_dic[key] = y_vec_spline

	return x_vec_spline_dic,y_vec_spline_dic

#create spline derivative
def create_dspline_dictionary(x_vec_spline_dic,y_vec_spline_dic):
	print('creating spline derivatives...')
	x_vec_dspline_dic = {}
	for key in x_vec_spline_dic:
		x_vec_dspline_dic[key] = x_vec_spline_dic[key].derivative()

	y_vec_dspline_dic = {}
	for key in y_vec_spline_dic:
		y_vec_dspline_dic[key] = y_vec_spline_dic[key].derivative()

	return x_vec_dspline_dic,y_vec_dspline_dic

#determine wave direction
def wave_direction(phase_snap,x_vec_dspline_dic,y_vec_dspline_dic,delta):
	print('calculating vector grid...')
	ydim, xdim = phase_snap.shape
	grid_y_dim = len(range(0,ydim,delta))
	grid_x_dim = len(range(0,xdim,delta))
	grid = np.zeros([grid_y_dim,grid_x_dim],dtype=complex)
	#fill in grid with (dphi/dx + dphi/dy *j)
	for y in range(0,ydim,delta):
		for x in range(0,xdim,delta):
			grid[int(y/delta),int(x/delta)] = complex(-x_vec_dspline_dic[y](x),-y_vec_dspline_dic[x](y))
	quiver_grid = grid/abs(grid)	#quiver_grid has vectors of magnitude=1
	
	return grid,quiver_grid

def compute_snap_to_quiver(phase_snap,delta):
	x_vec_dic,y_vec_dic = create_vector_dictionary(phase_snap,delta)
	x_vec_spline_dic,y_vec_spline_dic = create_spline_dictionary(x_vec_dic,y_vec_dic,smoothing=0.5,k_value=3)
	x_vec_dspline_dic,y_vec_dspline_dic = create_dspline_dictionary(x_vec_spline_dic,y_vec_spline_dic)
	grid,quiver_grid = wave_direction(phase_snap,x_vec_dspline_dic,y_vec_dspline_dic,delta)
	return grid,quiver_grid

def figsave_to_tif(string):
	wdir = os.getcwd()
	print("Working in",wdir)

	p = Path(wdir)
	movie_names = list( p.glob(str(string)+'*.tif' ))
	if len(movie_names) == 0:
	    print('Found no input movie.. exiting!')
	    sys.exit(1)

	movie_path = os.path.join( wdir, movie_names[0].name )
	rm = io.imread(movie_path, plugin="tifffile")

	mat = np.zeros([len(movie_names),rm.shape[0],rm.shape[1],rm.shape[2]], dtype = np.uint8)

	for index,k in enumerate(movie_names):

		movie_name = k.name # the roi movie file name

		movie_path = os.path.join( wdir, movie_name )
		print('Opening :', movie_name)
		mat[index,:,:,:] = io.imread(movie_path, plugin="tifffile")
		os.remove(movie_path)

	out_path1 = os.path.join(wdir, 'concat_'+string[:-1]+'.tif')
	io.imsave(out_path1, mat)
	print('written',out_path1)