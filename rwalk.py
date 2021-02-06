import numpy as np
import matplotlib.pyplot as plt


nsteps = 100
plt.axis([0, 100, -2, 2])
plt.title('Random walk model')
plt.xlabel('x-position')
plt.ylabel('z-position') 

cols = ['r','g','b','k','y']
lbls = ['Roger', 'Roderick', 'Ruben', 'Reginald', 'Brian']

for p in range(0,5):
    oldw = 0.0 
    for i in range(1, nsteps):
        yi = np.random.normal(0)
        # Weiner process
        neww = oldw+(yi/np.sqrt(nsteps))
        x = [i-1,i]
        y = [oldw, neww]
        #plt.scatter(i, neww, s=20, c=cols[p])
        plt.plot(x,y, c=cols[p])
        plt.pause(0.01)
        oldw = neww
    plt.plot(x,y, c=cols[p], label=lbls[p])
    plt.legend(loc='upper left', fontsize='x-small')
plt.show()