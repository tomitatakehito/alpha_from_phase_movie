import os,sys
from os import path,walk
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wavelet_ana_lib import *
from skimage import io
from scipy.interpolate import UnivariateSpline as spline
import math


def vector_field(dx_values,dy_values,prm,Nframes,movie_name,p):

    vector_movie = np.zeros( rm.shape,dtype = np.float32 ) # initialize empty array for output
    velocity_movie = np.zeros( rm.shape,dtype = np.float32 )
    quiver_movie = np.zeros( rm.shape,dtype = np.float32 )

    for x in p:
        
        print(x)
        
        for y in p:

            for t in np.arange(Nframes):

                direction = np.array([-dx_values[t,int(y/delta),int(x/delta)],-dy_values[t,int(y/delta),int(x/delta)]])
                dphi_dv = math.sqrt(dx_values[t,int(y/delta),int(x/delta)]**2+dy_values[t,int(y/delta),int(x/delta)]**2)

                mag_vec = 10*(2*np.pi/prm[t,y,x])/dphi_dv
                velocity_movie[t,y,x]=mag_vec
                
                origin = np.array([y,x])
                vector = origin + mag_vec*direction/dphi_dv
                quiver = origin + 1.4*delta*direction/dphi_dv

                #if mag_vec <30:
                #    for k in range(int(mag_vec)):
                #        dot = ((mag_vec-k)*origin+k*vector)/mag_vec
                #        if 0<int(round(dot[0]))<rm.shape[1]-1 and 0<int(round(dot[1]))<rm.shape[1]-1:
                #            vector_movie[t,int(round(dot[0])),int(round(dot[1]))]+=k**2

                #for k in range(int(1.4*delta)):
                #    dot = ((1.4*delta-k)*origin+k*quiver)/(1.4*delta)
                #    if 0<int(round(dot[0]))<rm.shape[1]-1 and 0<int(round(dot[1]))<rm.shape[1]-1:
                #        quiver_movie[t,int(round(dot[0])),int(round(dot[1]))]+=k**2



    #out_path1 = os.path.join(wdir, 'wave_vector_' + movie_name)
    #io.imsave(out_path1, vector_movie)
    #out_path2 = os.path.join(wdir, 'wave_velocity_' + movie_name)
    #io.imsave(out_path2, velocity_movie)
    #out_path3 = os.path.join(wdir, 'wave_quiver_' + movie_name)
    #io.imsave(out_path3, quiver_movie)



#---- to be overwritten by the prepare script----
dt = 5
Tmin = 100
Tmax = 220
nT = 100
#------------------------------------------------

delta = 1


wdir = os.getcwd()
print("Working in",wdir)

# None needed atm
# if len(sys.argv) < 2:
#     print("No command line argument.. exiting")
#     sys.exit(1)

p = Path(wdir)
movie_names = list( p.glob('phase*.tif') )
if len(movie_names) == 0:
    print('Found no input movie.. exiting!')
    sys.exit(1)

print("Found {} input movie(s):".format(len(movie_names), wdir))
print(movie_names)

for k in movie_names:

    movie_name = k.name # the roi movie file name
    
    #open phase movie as rm
    movie_path = os.path.join( wdir, movie_name )
    print('Opening :', movie_name)
    rm = io.imread(movie_path, plugin="tifffile")

    # open period movie as prm
    period_movie_name = "period_" + k.name[6:]
    period_movie_path = os.path.join( wdir, period_movie_name )
    print('Opening :', period_movie_name)
    prm = io.imread(period_movie_path, plugin="tifffile")

    #---------------------------------------------------------
    periods = np.linspace(Tmin,Tmax,nT)
    T_c = Tmax

    Nframes, ydim, xdim = rm.shape
    Npixels = ydim*xdim
    # not working, Fiji can't read this :/
    #wm = np.zeros( (*rm.shape,3),dtype = np.float32 ) # initialize empty array for output

    print( 'Computing the transforms for {} pixels'.format(Npixels) )
    sys.stdout.flush()

    space = np.linspace(0,rm.shape[1]-1,num=rm.shape[1]) # make array corresponding to xy dim of image
    p = range(0,rm.shape[1]-1,delta) # make array for eval_grid

    dx_values = np.zeros([Nframes,len(p),len(p)]) # initialize empty array for dx_values
    dy_values = np.zeros([Nframes,len(p),len(p)]) 
    vecx_values = np.zeros([Nframes,len(p),len(p)])
    vecy_values = np.zeros([Nframes,len(p),len(p)])
    # derive necessary dphase/dx, dphase/dy values

    for t in range(Nframes):

        print('t=',t)

        for k in p:
            x_vec = np.unwrap(rm[t,:,k])
            x_vec_spline = spline(space,x_vec, s = 0.5, k = 3)
            x_vec_spline1 = x_vec_spline.derivative()
            y_vec = np.unwrap(rm[t,k,:])
            y_vec_spline = spline(space,y_vec, s = 0.5, k = 3)
            y_vec_spline1 = y_vec_spline.derivative()
            
            for l in p:
                dx_values[t,int(l/delta),int(k/delta)] = x_vec_spline1(l)
                dy_values[t,int(k/delta),int(l/delta)] = y_vec_spline1(l)
                #if abs((2*np.pi/prm[t,l,k])/x_vec_spline1(l))<100:
                #    vecx_values[t,int(l/delta),int(k/delta)] = (2*np.pi/prm[t,l,k])/x_vec_spline1(l)
                #if abs((2*np.pi/prm[t,k,l])/y_vec_spline1(l))<100:
                #    vecy_values[t,int(k/delta),int(l/delta)] = (2*np.pi/prm[t,k,l])/y_vec_spline1(l)



    #vector_field(dx_values,dy_values,prm,Nframes,movie_name,p)  # output vector field visualisasion

    for x in p:
        
        print(x)
        
        for y in p:

            for t in np.arange(Nframes):

                direction = np.array([-dx_values[t,int(y/delta),int(x/delta)],-dy_values[t,int(y/delta),int(x/delta)]])
                dphi_dv = math.sqrt(dx_values[t,int(y/delta),int(x/delta)]**2+dy_values[t,int(y/delta),int(x/delta)]**2)

                mag_vec = 10*(2*np.pi/prm[t,y,x])/dphi_dv
                
                if mag_vec < 30:
                    vector = mag_vec*direction/dphi_dv
                    vecx_values[t,int(y/delta),int(x/delta)] = vector[0]
                    vecy_values[t,int(y/delta),int(x/delta)] = vector[1]

    # flux analysis


    for rad in range (1,50):
        radius = rad  # radius for surface to calculate flux, must be integer

        # make phi matrix according to len(p)
        phi_all = np.zeros([Nframes,len(p),len(p)],dtype=np.float32)
        

        # make P matrix according to radius
        E = np.eye(len(p))
        P = E
        for ra in range(2,radius+1):
            P += np.eye(len(p),k = ra-1) + np.eye(len(p),k = -(ra-1))

        P = np.matrix(P)


        # loop over time

        for t in range(Nframes):
            phi = np.matrix(np.zeros([len(p),len(p)])) # initialize phi 

            print('t=',t)
            VxP = np.matrix(vecx_values[t,:,:]) * P
            PVy = P * np.matrix(vecy_values[t,:,:])

            

            phi[:len(p)-radius,:] += -VxP[radius:,:]
            phi[radius:,:] += VxP[:len(p)-radius,:]
            phi[:,:len(p)-radius] += -PVy[:,radius:]
            phi[:,radius:] += PVy[:,:len(p)-radius]

            phi_all[t,:,:] = phi

        out_path4 = os.path.join(wdir, 'flux_' +str(rad)+ movie_name)
        io.imsave(out_path4, phi_all)







        


#        for k in p:
#            y_vec = np.unwrap(rm[t,k,:])
#            y_vec_spline = spline(space,y_vec, s = 0.5, k = 3)
#            y_vec_spline1 = y_vec_spline.derivative()

#            for l in p:
#                dy_values[t,int(k/delta),int(l/delta)] = y_vec_spline1(l)