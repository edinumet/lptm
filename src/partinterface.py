"""
Created: Tuesday 1st December 2020
@author: John Moncrieff (j.moncrieff@ed.ac.uk)
Last Modified on 5 Feb 2021 16:30 

DESCRIPTION
===========
This package contains the class object for configuring and running 
the PARTICLE Jupyter notebook

"""
import os
import subprocess
import ipywidgets as widgets
from IPython.display import display

class partinterface():
    def __init__(self):
        # ADMS stability classes
        self.stability = {
            'A':  {'u*': 0.1, 'H': 350, "L":-2 },
            'B':  {'u*': 0.2, 'H': 250, "L":-10 },
            'C':  {'u*': 0.5, 'H': 150, "L":-100 },
            'D':  {'u*': 0.5, 'H': 50, "L":0 },
            'E':  {'u*': 0.3, 'H': 25, "L":100 },
            'F':  {'u*': 0.2, 'H': -5, "L":20 },
            'G':  {'u*': 0.1, 'H': -25, "L":5 },
            }

        self.dd_stab = widgets.Dropdown(
                                         options=['A', 'B', 'C', 'D','E','F','G'],
                                         value = 'D', 
                                         description="Pasquill stability", width=50
                                         )
        self.dd_stab.observe(self.on_change,names='value')
        self.stab = self.dd_stab.value
        self.btn = widgets.Button(description='Run Model', width=100)
        self.btn.style.button_color = 'tomato'
        self.btn.on_click(self.btn_eventhandler)

        self.h1 = widgets.HBox(children=[self.dd_stab]) #,self.btn])


    def on_change(self,change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.stab=change['new']
                print("changed to %s" % change['new'])
    
    def btn_eventhandler(self,obj):
            print("Run complete - goto next cell")    
        
        
