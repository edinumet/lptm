# Simple test of Obukhov length routine

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

Tv = 293     # Kelvin
ustar = 0.25  # m s-1
x = []
y = []

# H in W m-2 since it'll be converted
# to kinematic units later
# Loop around H
H_limits = [i for i in np.linspace(-55, 350, 18)]

for H in H_limits:
        L = (-1*Tv * ustar**3)/(0.4*9.83 * H/1200)
        x.append(H)
        y.append(L)
#print(y)

y_interp = scipy.interpolate.interp1d(x, y)
#print(y_interp(-50.0))

# Plot the results
fig = plt.figure(figsize=(6,4))
ax = plt.subplot(1, 1, 1)
ax.plot(x, y, c='r')
# Loop over axes to find stability regimes
j = 0
# for i in H_limits:
#     if i < 0 and y[j] < 90:  # strongly stable
#         plt.vlines(x = i, ymin=-400, ymax=400)
#     elif i < 0 and 90 <y[j] < 130:  # stable
#         plt.vlines(x = i, ymin=-400, ymax=400)
#     elif i < 0 and y[j] > 130:      # slightly stable
#         plt.vlines(x = i, ymin=-400, ymax=400)
#     elif i > 0 and -650 < y[j] < -10000:      # neutral
#         plt.vlines(x = i, ymin=-400, ymax=400)
#     elif i > 0 and -30 > y[j] > -650:     # slightly convective
#         plt.vlines(x = i, ymin=-400, ymax=400)
#     elif i > 0 and -5.001> y[j] > -30:     # convective
#         plt.vlines(x = i, ymin=-400, ymax=400)
#     elif i > 0 and -0.001 > y[j] > -5:   # strongly convective
#         plt.vlines(x = i, ymin=-400, ymax=400)   
#     j = j+1


# highlight a time range 
#ax.axvspan(2005, 2010, color="green", alpha=0.6) 

ax.set_xlabel('Sensible Heat Flux ($W \ m^{-2}$)')
ax.set_ylabel('Obukhov Length (m)')
#ax3.text(0.1, 0.9, r's1 = %s $ng m^{-3}$' % '{0:.2f}'.format(sites[0]), fontsize=9,color='red')
ax.set_title('Obukhov length as a function of Sensible Heat Flux \n $u_*$ = {}'.format(ustar) + ' $m \ s^{-1}$,' +' $\ T_v$ = {}'.format(Tv) +' K', fontsize=11, color='k')
plt.savefig('Obukhov_length.png')
plt.show()

