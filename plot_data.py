import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as spline

data = np.load('fnumpy.npy')

abs_data = abs(data)
print(abs_data.shape)
vector = abs_data[:,6,4]
time_vector = np.arange(len(vector))*10

vec_spline = spline(time_vector,vector, s = 0.002, k = 3)
vec_spline_der = vec_spline.derivative()
alpha_vector = vec_spline_der(time_vector)/vec_spline(time_vector)

plt.plot(time_vector,abs_data[:,6,4],alpha = 0.3,color = 'c')
plt.plot(time_vector,vec_spline(time_vector),alpha = 0.3,color = 'm')
plt.plot(time_vector,vec_spline_der(time_vector),alpha = 0.3,color = 'r')
plt.plot(time_vector,alpha_vector,alpha = 0.3,color = 'g')

plt.ylim([-0.002,0.07])
plt.show()