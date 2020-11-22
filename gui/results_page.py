from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image

import pandas as pd 
import numpy as np

class Results(BoxLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root

        anchor1 = AnchorLayout(anchor_x='left', anchor_y='bottom')
        home_button = Button(text="Home", size_hint=(0.2, .1))
        home_button.bind(on_press=self.go_home)
        anchor1.add_widget(home_button)
        self.add_widget(anchor1)

        df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'])
        dfs = df.to_string()
        f = open("image_path.txt", "r")
        image_path = f.read()

        anchor2 = AnchorLayout(anchor_x='center', anchor_y='top')
        about_text = Label()
        about_text.text = dfs
        anchor2.add_widget(about_text)
        self.add_widget(anchor2)

        anchor3 = AnchorLayout(anchor_x='center', anchor_y='center')
        pro_image = Image(source='capture_20201120-224554.png')
        anchor3.add_widget(pro_image)
        self.add_widget(anchor3)
    
    def go_home(self, instance):
        self.sn_root.go_home()