from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import os
import time

class Capture(BoxLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root

        self.home_button = Button(text="Home")
        self.home_button.bind(on_press=self.go_home)
        self.add_widget(self.home_button)

        self.cameraObject = Camera(play=False)
        self.cameraObject.play = True
        self.cameraObject.resolution = (300, 300)
       
        self.cameraClick = Button(text="Take Photo")
        self.cameraClick.size_hint=(.5, .2)

        self.cameraClick.bind(on_press=self.onCameraClick)

        self.add_widget(self.cameraObject)
        self.add_widget(self.cameraClick)
   
    def onCameraClick(self, *args):
        timest = time.strftime("%Y%m%d-%H%M%S")
        self.cameraObject.export_to_png(os.path.join("D:/Git/SolveNet/captures", ("capture_"+timest+".png")))

    def go_home(self, instance):
        self.sn_root.go_home()
