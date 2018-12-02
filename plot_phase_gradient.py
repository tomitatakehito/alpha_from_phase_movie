import os,sys
from os import path,walk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from skimage import io
import math
import alpha_library as al

#set values
#----------------------------------------------
t = 100		#time point of the movie to process
delta = 2	#fineness of the grid
radius = 20	#radius for flux calculation
#----------------------------------------------

wdir = os.getcwd()
print("Working in",wdir)

p = Path(wdir)
movie_names = list( p.glob('phase*.tif') )

movie_name = movie_names[0]

phase_movie = al.read_movie(wdir,movie_name)
ydim, xdim = phase_movie.shape[1:]
grid_y_dim = len(range(0,ydim,delta))
grid_x_dim = len(range(0,xdim,delta))
grid_movie = np.zeros([phase_movie.shape[0],grid_y_dim,grid_x_dim],dtype=complex)

for t in range(phase_movie.shape[0])[:1]:
	print("Working on",str(t),"/",str(phase_movie.shape[0]),"frame")

	phase_snap = phase_movie[t,:,:]

	grid, quiver_grid = al.compute_snap_to_quiver(phase_snap,delta)
	grid_movie[t,:,:] = grid

# for y in range(grid_y_dim):
# 	for x in range(grid_x_dim):
# 		vector = grid_movie[:,y,x]
# 		plt.plot(abs(vector))
# plt.ylim([0,0.5])
# plt.show()

np.save('fnumpy.npy', grid_movie)