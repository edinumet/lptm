import numpy as np
import matplotlib.pyplot as plt

class Model():
    # self.model = Model(1000, 0, 0.3, 30, 500, 1000)
    def __init__(self, BL_height, disp_height, roughness, Tv, max_range, max_height):
        # sets initial conditions - assumed constant over the entire domain
        self.x = None
        self.BL_height = BL_height
        self.disp_height = disp_height
        self.Tv = Tv
        self.roughness = roughness
        self.xrange = max_range
        self.max_height = max_height
        self.exit_count = np.zeros([360, 300])
        self.xi, self.yi = np.mgrid[0:360:1, 0:300:1]
        self.yi = self.yi / 3
        self.OL = None
        self.figure = plt.figure()
        self.axes1 = self.figure.add_subplot(111, projection='3d')
        self.axes1.set_xlim(-100, 0)
        self.axes1.set_ylim(-200, 200)
        self.axes1.set_zlim(0, 100)
        self.axes1.set_yticks([])
        #self.axes2 = self.figure.add_subplot(212)
        #self.plt.ion()

    def add_particles(self, count, x, y, z, ustar, theta, H):
        #  This function adds particles to the set of all particles used in the diffusion calculations
        #  x, y and z are lists of either length 1 or 2  They define the range of the source area
        #        if all three are single values then the source is a point
        #        if one variable has two values then it is a line source
        #        if two variables have two values then it is an area source
        #        if all three variables have two values then it is a volume source
        #
        #  ustar, theta, and H are friction velocity, wind direction and sensible heat flux
        #       the set of particles retain the associated values during their diffusion
        #
        #  flg is a label attached to the group of particles created.
        #

        # get the x,y, and z cartesian domain limits
        if len(x) == 1:
            x0 = x
            x1 = x
        else:
            x0 = x[0]
            x1 = x[1]
        if len(y) == 1:
            y0 = y
            y1 = y
        else:
            y0 = y[0]
            y1 = y[1]
        if len(z) == 1:
            z0 = z
            z1 = z
        else:
            z0 = z[0]
            z1 = z[1]
        newx = np.random.uniform(x0, x1, count)
        newy = np.random.uniform(y0, y1, count)
        newz = np.random.uniform(z0, z1, count)

        theta = np.deg2rad(theta)
        # if no particles created yet then define the first set of parameters
        if self.x is None:
            self.x = newx
            self.y = newy
            self.z = newz
            self.up = np.zeros(count)
            self.vp = np.zeros(count) 
            self.wp = np.zeros(count) 
            self.ustar = np.ones(count) * ustar
            self.theta = np.ones(count) * theta
            self.H = np.ones(count) * H
            #self.flg = np.ones(count) * 1.0
        # else add new particles to existing set of particles    
        else:
            self.x = np.append(self.x, newx)
            self.y = np.append(self.y, newy)
            self.z = np.append(self.z, newz)
            self.up = np.append(self.up, np.zeros(count))
            self.vp = np.append(self.vp, np.zeros(count))
            self.wp = np.append(self.wp, np.zeros(count))
            self.ustar = np.append(self.ustar, np.ones(count) * ustar)
            self.theta = np.append(self.theta, np.ones(count) * theta)
            self.H = np.append(self.H, np.ones(count) * H)
            #self.flg = np.append(self.flg, np.ones(count) .dot(flg))

    def limit_to_range(self):
        dist = np.sqrt(self.x**2 + self.y**2)
        
        # tally particles outside domain
        idout = dist >= self.xrange
        if np.sum(idout)>0:
            direct = np.round(np.rad2deg(np.arctan2(self.y[idout], self.x[idout]))).astype(int)
            direct[direct<0] = direct[direct<0] + 360
            height = np.round(self.z[idout]*300/self.max_height).astype(int)
            height[height > 299] = 299
            self.exit_count[direct,height] =  self.exit_count[direct,height] + 1
        
        # eliminate particles outside domain
        idin = dist < self.xrange
        self.x = self.x[idin]
        self.y = self.y[idin]
        self.z = self.z[idin]
        self.up =self.up[idin]
        self.vp = self.vp[idin]
        self.wp = self.wp[idin]
        self.ustar = self.ustar[idin]
        self.theta = self.theta[idin]
        self.H = self.H[idin]
        #self.flg = self.flg[idin]

    def update_particles(self, dt):
        L = (-1 * self.Tv * self.ustar**3)/(0.4 * 9.83 * self.H/1200)
        self.OL = L[0]
        zmd = self.z - self.disp_height
        ids = L >= 0.0
        idn = np.abs(L) > 99.0
        idu = L < 0.0

        sigma_u = np.zeros(len(self.x))
        sigma_v = np.zeros(len(self.x))
        sigma_w = np.zeros(len(self.x))
        T_u = np.zeros(len(self.x))
        T_v = np.zeros(len(self.x))
        T_w = np.zeros(len(self.x))
        psi = np.zeros(len(self.x))
        psi_0 = np.zeros(len(self.x))
        
        # standard deviations
        sigma_u[ids] = 2.0*self.ustar[ids]
        sigma_v[ids] = 1.4*self.ustar[ids]
        sigma_w[ids] = 1.25*self.ustar[ids]
        
        sigma_u[idu] = np.sqrt(0.35*self.ustar[idu]**2*((1000/(0.4*L[idu]))**2)**(1/3) + 2.0*self.ustar[idu]**2)
        sigma_v[idu] = sigma_u[idu]
        sigma_w[idu] = 1.3*self.ustar[idu] * (1.0 - 3*zmd[idu]/L[idu])**(1/3)
        
        # Lagrangian  time scales
        T_u = 0.5*zmd/sigma_u
        T_v = 0.5*zmd/sigma_v
        T_w = 0.5*zmd/sigma_w
        
        T_u[ids] = T_u[ids]*(1+5*zmd[ids]*L[ids])**-1
        T_v[ids] = T_v[ids]*(1+5*zmd[ids]*L[ids])**-1
        T_w[ids] = T_w[ids]*(1+5*zmd[ids]*L[ids])**-1
        
        T_u[idu] = T_u[idu] *(1-6*zmd[idu]/L[idu])**0.25
        T_v[idu] = T_v[idu] *(1-6*zmd[idu]/L[idu])**0.25
        T_w[idu] = T_w[idu] *(1-6*zmd[idu]/L[idu])**0.25
        
        # wind speed extracted from stability adjusted profile
        x1 = (1-15*zmd[idu]/L[idu])**0.25
        psi[idu] = 2*np.log( (1+x1)/2) + np.log( (1+x1**2)/2) - 2*np.arctan(x1)+np.pi/2
        psi[idn] = 0
        psi[ids] = -4.7*zmd[ids]/L[ids]

        x2 = (1-15*self.roughness/L[idu])**0.25
        psi_0[idu] = 2*np.log( (1+x2)/2) + np.log( (1+x2**2)/2) - 2*np.arctan(x2)+np.pi/2
        psi_0[idn] = 0
        psi_0[ids] = -4.7*self.roughness/L[ids]

        Uh = (self.ustar/0.4) * (np.log ( zmd/ self.roughness ) - (psi - psi_0))
    
        # calculate the random walk/markov chain fluctuating velocity components
        self.up = self.up*(1-dt/T_u)+(2*dt*sigma_u**2/T_u)**0.5* np.random.randn(len(sigma_u))
        self.vp = self.vp*(1-dt/T_v)+(2*dt*sigma_v**2/T_v)**0.5* np.random.randn(len(sigma_v))
        self.wp = self.wp*(1-dt/T_w)+(2*dt*sigma_w**2/T_w)**0.5* np.random.randn(len(sigma_w))
        
        # calc mean horizontal wind vector components 
        uh = Uh*np.cos(self.theta)
        vh = Uh*np.sin(self.theta)

        # advect the particle by the mean + fluctuation over the time step
        self.x = self.x + dt*(uh+self.up) 
        self.y = self.y + dt*(vh+self.vp) 
        self.z = np.abs(self.z + dt*self.wp)
        
        # tally then eliminate particles outwith the measurement domain
        self.limit_to_range()
        
    def redraw_plot(self, ix, xi, yi, x, y, z, exit_count, OL):
        # plot first 300 time steps then update every 500 time steps
        if ix < 300:
            self.axes1.cla()
            self.axes1.scatter(x, y, z, s=1, c='red')
            
            self.axes1.set_xlim(-100, 0)
            self.axes1.set_ylim(-200, 200)
            self.axes1.set_zlim(0, 100)
            self.axes1.set_yticks([])
            elev = 0.0
            azim = 90.0
            self.axes1.view_init(elev, azim)
            #plt.show()
            #self.axes2.contourf(xi, yi, exit_count, 20, cmap=plt.cm.rainbow)
            plt.suptitle('time, sec: '+f"{ix*0.1:9.1f}"+'\n'+'Stability = '+f"{OL:9.1f}")
            plt.pause(0.05)
            self.figure.canvas.draw()
            #plt.ion()
            #plt.draw()
        elif np.mod(ix, 500) == 0:
            self.axes1.cla()
            self.axes1.scatter(x, y, z, s=1, c='red')
            
            self.axes1.set_xlim(-220, 0)
            self.axes1.set_ylim(-200, 200)
            self.axes1.set_zlim(0, 100)
            #plt.show()
            #self.axes2.contourf(xi, yi, exit_count, 20, cmap=plt.cm.rainbow)
            plt.suptitle('time: (s):'+(f"{ix*0.1:9.1f}"))
            plt.pause(0.05)
            self.figure.canvas.draw()
    
    def run(self):
        # loop over time (0.2 sec time step)
        for self.ix in range(300):
            # update particles
            self.update_particles(0.2)
            self.redraw_plot(self.ix, self.xi, self.yi, self.x, self.y, self.z, self.exit_count, self.OL)
            # add {count} number of new particles at each time step
            # (self, count, x, y, z, ustar, theta, H)
            self.add_particles(50, [0], [0], [40], 0.2, -180, 150)