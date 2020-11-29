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

        #dfs = df.to_string()
        #f = open("image_path.txt", "r")
        #image_path = f.read()

        self.ans_text = Label()
        #self.ans_text.text = ans
        

        self.df_text = Label()
        #self.df_text.text = dfs
        #self.df_text.text = "A"
        

        #self.pro_image = Image(source=r'D:\123.png')
        self.pro_image = Image()
        

        self.home_button = Button(text="Home", size_hint=(0.3, 0.3))
        self.home_button.bind(on_press=self.go_home)

        

    def go_home(self, instance):
        self.sn_root.go_home()

    def print_results(self, direc):
        df, ans, img = self.sn_root.rrun(img_path=direc)
        dfs = df.to_string()
        self.ans_text.text = ans
        self.df_text.text = dfs
        self.pro_image.source = img
        self.add_widget(self.ans_text)
        self.add_widget(self.df_text)
        self.add_widget(self.pro_image)
        self.add_widget(self.home_button)
       