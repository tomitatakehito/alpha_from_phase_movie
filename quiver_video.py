import os,sys
from os import path,walk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from skimage import io
from scipy.interpolate import UnivariateSpline as spline
import math
import alpha_library as al

#set values
#----------------------------------------------
t = 100		#time point of the movie to process
delta = 50	#fineness of the grid
radius = 20	#radius for flux calculation
#----------------------------------------------

wdir = os.getcwd()
print("Working in",wdir)

p = Path(wdir)
movie_names = list( p.glob('phase*.tif') )

print(movie_names)

def analyse(movie_name):

	phase_movie = al.read_movie(wdir,movie_name)

	for t in range(phase_movie.shape[0]):

		phase_snap = phase_movie[t,:,:]

		grid, quiver_grid = al.compute_snap_to_quiver(phase_snap,delta)

		r,c = grid.shape
		R,C = np.meshgrid(np.arange(0,r),np.arange(0,c))
		U = np.imag(grid[R,C])
		V = np.real(grid[R,C])
		fig = plt.figure(figsize = [8,8])
		Q = plt.quiver(R,C, U, V, units='width')
		# qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')

		# fig = plt.figure()
		# ax = fig.gca()
		# ax.imshow(flux)

		#plt.show()
		fig.savefig('quiver'+str(t+1000)+'.tif')
		

		# out_path1 = os.path.join(wdir, 'flux_'+str(t)+'.tif')
		# io.imsave(out_path1, flux)
		# print('written',out_path1)