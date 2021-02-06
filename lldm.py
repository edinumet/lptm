# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:33:15 2017
Last Edited 18 Dec 2018 23:38
@author: rclement
         jbm modified it to Model-View-Controller format and made it more suitable
         for undergraduate teaching by adding scrollbars etc
"""
import numpy as np
import wx

#http://wiki.wxpython.org/WxLibPubSub
# pubsub changed with latest wxPython
if "2.8" in wx.version():
    import wx.lib.pubsub.setupkwargs
    from wx.lib.pubsub import pub
else:
    from wx.lib.pubsub import pub
import wx.aui
import sys
import os
import numpy as np
from wx.lib.wordwrap import wordwrap
import matplotlib as mpl
if os.name == 'posix':
    mpl.use('WXAgg')  # otherwise gets a segmentation fault under Linux
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Model():
    def __init__(self, BL_height, disp_height, roughness, Tv, max_range, max_height):
        # sets initial conditions - assumed constant over the entire domain
        self.x = None
        self.BL_height = BL_height
        self.disp_height = disp_height
        self.Tv = Tv
        self.roughness = roughness
        self.range = max_range
        self.max_height = max_height
        self.exit_count = np.zeros([360,300])
        self.xi, self.yi = np.mgrid[0:360:1, 0:300:1]
        self.yi = self.yi / 3
        self.height_total = []

        # add initial particles
        #flg = 1
        self.add_particles(15, [0], [0], [40], 0.8, 180, 150)

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
        idout = dist >= self.range
        if np.sum(idout)>0:
            direct = np.round(np.rad2deg(np.arctan2(self.y[idout], self.x[idout]))).astype(int)
            direct[direct<0] = direct[direct<0] + 360
            height = np.round(self.z[idout]*300/self.max_height).astype(int)
            height[height > 299] = 299
            self.exit_count[direct,height] =  self.exit_count[direct,height] + 1
            #print("direct = {}, height = {}".format(direct,height))
            self.height_total.append(height.tolist())
        
        # eliminate particles outside domain
        idin = dist < self.range
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

    def update_particles(self, dt, nt):
        L = (-1 * self.Tv * self.ustar**3)/(0.4 * 9.83 * self.H/1200)
        if nt ==0:
            print("MOL = {}, H = {}, ustar = {}, Tv = {}".format(L, self.H, self.ustar, self.Tv))
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
        
        T_u[ids] = T_u[ids]*(1+5*zmd[ids]*L[ids])**-1    # LizH 4.14
        T_v[ids] = T_v[ids]*(1+5*zmd[ids]*L[ids])**-1
        T_w[ids] = T_w[ids]*(1+5*zmd[ids]*L[ids])**-1
        
        T_u[idu] = T_u[idu] *(1-6*zmd[idu]/L[idu])**0.25    # LizH 4.15
        T_v[idu] = T_v[idu] *(1-6*zmd[idu]/L[idu])**0.25
        T_w[idu] = T_w[idu] *(1-6*zmd[idu]/L[idu])**0.25
        
        # wind speed extracted  from stability adjusted profile
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

    def plot_histogram(self,data):
        print("reached plot histo")
        bins = np.arange(10,300,5)
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(1, 1, 1)
        plt.hist(data, bins=bins, alpha=0.5)
        plt.show()

    def run(self):
        # loop over time (0.1 sec time step)
        #print('Reached model run function')
        nt = 0
        for self.ix in range(600):
            # update particles
            #print(self.ix)
            self.update_particles(0.1, nt)
            nt = nt+1
            # def redraw_plot(self, ix, xi, yi, x, y, z, exit_count, flg)
            pub.sendMessage("PART.CHANGED", value1=self.ix, value2=self.xi, value3=self.yi, value4=self.x,
                            value5=self.y, value6=self.z, value7=self.exit_count)
            #### 9999 send particle info
            # add new particles at each time step
            self.add_particles(15, [0], [0], [40], 0.8, 180, 150)
        
        self.flatList=[]
        for elem in self.height_total:   # convert list of lists to a flat list
            self.flatList.extend(elem)
        #print(self.flatList)    
        self.plot_histogram(self.flatList)
        

class View(wx.Frame):
    # from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
    """The main frame of the application
    """

    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None)
        self.SurfaceType = ["grass (dry)", "cereals (dry)", "conifers (dry)",
                        "bare soil (dry)", "water", "upland", "grass (wet)", "conifers (wet)"]
        self.create_menu()
        self.mainPanel = wx.Panel(self, -1, style=wx.RAISED_BORDER)

        self.top_panel = wx.Panel(self.mainPanel, -1, style=wx.RAISED_BORDER)
        self.top_left_panel = wx.Panel(self.top_panel, -1, style=wx.RAISED_BORDER)
        self.top_right_panel = wx.Panel(self.top_panel, -1, style=wx.RAISED_BORDER)
        self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.top_sizer.Add(self.top_left_panel, -1, wx.EXPAND | wx.ALL, border=10)
        self.top_sizer.Add(self.top_right_panel, -1, wx.EXPAND | wx.ALL, border=10)

        # bottom panel ie the results of the calculation and below the figure
        self.bottom_panel = wx.Panel(self.mainPanel, -1, style=wx.RAISED_BORDER)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        # self.Maximize(True)
        self.btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vboxCP = wx.BoxSizer(wx.VERTICAL)

        self.top_left_panel.figure = plt.figure()
        # self.top_left_panel.figure.tight_layout()

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.2  # the amount of height reserved for white space between subplots
        #self.top_left_panel.canvas = FigureCanvas(self.top_left_panel, -1, self.top_left_panel.figure)
        self.axes1 = self.top_left_panel.figure.add_subplot(111, projection='3d')
        #self.axes2 = self.top_left_panel.figure.add_subplot(212)
        #self.axes3 = self.top_left_panel.figure.add_subplot(313)
        plt.ion()

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vboxOP = wx.BoxSizer(wx.VERTICAL)
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hboxCB = wx.BoxSizer(wx.HORIZONTAL)

        # Right-Hand CONTROL PANEL #

        self.gridSizer = wx.GridSizer(rows=5, cols=1, hgap=5, vgap=5)
        self.vboxCP = wx.BoxSizer(wx.VERTICAL)
        self.runBtn = wx.Button(self.top_right_panel, wx.ID_ANY, 'Run')
        self.quitBtn = wx.Button(self.top_right_panel, wx.ID_ANY, 'Quit')
        self.Bind(wx.EVT_BUTTON, self.onRUN, self.runBtn)
        self.Bind(wx.EVT_BUTTON, self.onQuit, self.quitBtn)

        self.vbox.Add(self.top_left_panel, -1, wx.EXPAND | wx.ALL, 5)  # proportion 2
        self.vbox.Add(self.top_right_panel, -1, wx.EXPAND | wx.ALL, 5)  # proportion 1
        self.hbox.Add(self.bottom_panel, -1, wx.EXPAND | wx.ALL, 5)  # proportion 1
        self.btnSizer.Add(self.runBtn, 0, wx.ALL, 5)
        self.btnSizer.Add(self.quitBtn, 0, wx.ALL, 5)
        self.vboxCP.Add(self.btnSizer, 0, wx.ALL | wx.CENTER, 5)
        self.top_right_panel.SetSizer(self.vboxCP)

        self.SetSize((910, 600))
        self.SetTitle('Lagrangian Particle Transport Model')
        self.top_panel.SetSizer(self.top_sizer)
        self.mainPanel.SetSizer(self.main_sizer)
        self.main_sizer.Add(self.top_panel, -1, wx.EXPAND | wx.ALL, border=10)
        self.main_sizer.Add(self.bottom_panel, -1, wx.EXPAND | wx.ALL, border=10)

        self.Centre()
        self.statusbar = self.CreateStatusBar()

    def redraw_plot(self, ix, xi, yi, x, y, z, exit_count):
        # plot first 200 time steps then update every 500 time steps
        if ix < 300:
            self.axes1.cla()
            self.axes1.scatter(x, y, z, s=1, c='red')
            self.axes1.set_xlim(-200, 200)
            self.axes1.set_ylim(-200, 200)
            self.axes1.set_zlim(0, 100)
            #plt.show()
            #ax2.contourf(xi, yi,  mod.exit_count, 20, cmap=plt.cm.rainbow)#,
            plt.suptitle('time, sec: '+str(ix*0.1))
            plt.pause(0.01)
        elif np.mod(ix, 500) == 0:
            self.axes1.cla()
            self.axes1.scatter(x, y, z, s=1, c='red')
            self.axes1.set_xlim(-200, 200)
            self.axes1.set_ylim(-200, 200)
            self.axes1.set_zlim(0, 100)
            #plt.show()
            #self.axes2.contourf(xi, yi, exit_count, 20, cmap=plt.cm.rainbow) #,
            plt.suptitle('time, sec: '+str(ix*0.1))
            plt.pause(0.01)


    def create_menu(self):
        menubar = wx.MenuBar()
        file = wx.Menu()
        edit = wx.Menu()
        help = wx.Menu()
        file.Append(101, '&Open', 'Open a new document')
        file.Append(102, '&Save', 'Save the document')
        file.AppendSeparator()
        quit = wx.MenuItem(file, 105, '&Quit\tCtrl+Q', 'Quit the Application')
        # quit.SetBitmap(wx.Image('stock_exit-16.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap())
        file.Append(quit)
        edit.Append(201, 'check item1', '', wx.ITEM_CHECK)
        edit.Append(202, 'check item2', kind=wx.ITEM_CHECK)
        submenu = wx.Menu()
        submenu.Append(301, 'radio item1', kind=wx.ITEM_RADIO)
        submenu.Append(302, 'radio item2', kind=wx.ITEM_RADIO)
        submenu.Append(303, 'radio item3', kind=wx.ITEM_RADIO)
        edit.Append(203, 'submenu', submenu)
        menubar.Append(file, '&File')
        menubar.Append(edit, '&Edit')
        menubar.Append(help, '&Help')
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.OnQuit, id=105)
        return

    def OnNew(self, event):
        self.statusbar.SetStatusText('New Command')

    def OnOpen(self, event):
        self.statusbar.SetStatusText('Open Command')

    def OnSave(self, event):
        self.statusbar.SetStatusText('Save Command')

    def OnExit(self, event):
        self.Close()

    def OnButton(self, event):
        """Called when the Reset Button is clicked"""
        event_id = event.GetId()
        event_obj = event.GetEventObject()
        if event_id == -2024:     # needs to be flexible 999999
            self.statusbar.SetStatusText("Reset Button Clicked")
            pub.sendMessage("RESET.CLICKED")
        print("ID=%d" % event_id)
        print("object=%s" % event_obj.GetLabel())

    def OnCombo(self, event):
        selection = self.cbx.GetStringSelection()
        index = self.cbx.GetSelection()
        print("Selected Item: %d '%s'" % (index, selection))

    def OnQuit(self, event):
        dlg = wx.MessageDialog(self, "Do you really want to close "
                                     "this application?",
                               "Confirm Exit",
                               wx.OK|wx.QUIT|wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.Destroy()
            sys.exit(0)

    def onAboutDlg(self, event):
        info = wx.AboutDialogInfo()
        info.Name = "particle lagrangian Transport Model"
        info.Version = "0.1"
        info.Copyright = "(C) 2018 School of GeoSciences"
        info.Description = wordwrap(
            "This is a lagrangian transport model written in Python "
            "(3.6) and wxPython (4.0)",
            350, wx.ClientDC(self))
        info.WebSite = ("http://www.geos.ed.ac.uk", "School Home Page")
        info.Developers = ["Robert Clement, John Moncrieff"]
        info.License = wordwrap("GNU Open Source", 500,
                            wx.ClientDC(self))
        # Show the wx.AboutBox
        wx.AboutBox(info)

    def OnAbout2(self, event):
        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
        dlg = wx.MessageDialog(self, "Particle Transport Model",
                               "About LPTM", wx.OK)
        dlg.ShowModal()  # Show it
        dlg.Destroy()  # finally destroy it when finished.

    def OnHowTo(self, event):
        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
        dlg = wx.MessageDialog(self, "Control is via the ComboBoxes ... ",
                               "About LPTM", wx.OK)
        dlg.ShowModal()  # Show it
        dlg.Destroy()  # finally destroy it when finished.

    def vibrate(self, win, count=20, delay=50):
        if count == 0:
            return
        x, y = win.GetPositionTuple()
        # print x,y
        dx = 2 * count * (.5 - count % 2)
        win.SetPosition((x+dx, y))
        wx.CallLater(delay, vibrate, win, count-1, delay)

    def OnCloseWindow(self, event):
        # self.Destroy()
        sys.exit(0)

    def onRUN(self, event):
         print('onRUN handler')
         pub.sendMessage('RUN.CLICKED')

    def onQuit(self, event):
        sys.exit(0)


class Controller:
    def __init__(self, app):
        # Create a new MODEL object
        # (self, BL_height, disp_height, roughness, Tv, max_range, max_height)
        self.model = Model(1000, 0, 0.3, 280, 200, 100)
        # Create a new VIEW object
        self.view = View()

        # This is the V3 model for PubSub
        # First parameter = the name of the Handler function doing the work
        # Second parameter = the name of the Message being broadcast
        pub.subscribe(self.ResetClicked, 'RESET.CLICKED')
        pub.subscribe(self.PARTChanged, 'PART.CHANGED')
        pub.subscribe(self.RunClicked, 'RUN.CLICKED')

        self.view.Show(True)

    def ResetClicked(self):
        """
        This method is the handler for "START CLICKED" messages,
        which pubsub will call as messages are sent from the view Start button.
        """
        print('Reset Button detected - resetting to default model')
        self.model.onReset()

    def RunClicked(self):
        """
        This method is the handler for "RUN.CLICKED" messages,
        which pubsub will call as messages are sent from the view RUN button.
        """
        print('Run Clicked and seen in Controller')
        # Create a new MODEL object
        # (self, BL_height, disp_height, roughness, Tv, max_range, max_height)
        self.model = Model(1000, 0, 0.3, 280, 200, 100)
        self.model.run()

    def PARTChanged(self, value1, value2, value3, value4, value5, value6, value7):
        """
        This method is the handler for "START CLICKED" messages,
        which pubsub will call as messages are sent from the view Start button.
        """
        #print('Particle position changed')
        # def redraw_plot(self, ix, xi, yi, x, y, z, exit_count, flg)
        self.view.redraw_plot(value1, value2, value3, value4, value5, value6, value7)


if __name__ == "__main__":
    app = wx.App(False)
    controller = Controller(app)

    # open a splash screen if it exists

    #if os.path.exists(SPLASH_SCREEN_FILENAME):
    #    splash_image = wx.Image(SPLASH_SCREEN_FILENAME, wx.BITMAP_TYPE_ANY, -1)
    #    wx.SplashScreen(splash_image.ConvertToBitmap(),
    #                    wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_TIMEOUT,
    #                    4000,
    #                    None, -1)
    app.MainLoop()
