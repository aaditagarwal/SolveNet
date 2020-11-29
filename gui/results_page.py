from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image

import pandas as pd 
import numpy as np

class Results(GridLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root
        self.cols = 2

        self.ans_text = Label(markup=True)
        
        self.df_text = Label()
        
        self.pro_image = Image()
        
        self.button_layout = AnchorLayout(anchor_x='left', anchor_y='bottom')
        home_button = Button(text="Home", size_hint=(0.3, .3))
        home_button.bind(on_press=self.go_home)
        self.button_layout.add_widget(home_button)
        
    def go_home(self, instance):
        self.sn_root.go_home()

    def print_results(self, direc):
        df, ans, img = self.sn_root.rrun(img_path=direc)
        dfs = df.to_string()
        self.ans_text.text = '[size=40][b]'+ans+'[/b][/size]'+"           "
        self.df_text.text = dfs
        self.pro_image.source = img
        self.add_widget(self.ans_text)
        self.add_widget(self.df_text)
        self.add_widget(self.button_layout)
        self.add_widget(self.pro_image)

       