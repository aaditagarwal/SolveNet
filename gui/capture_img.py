from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
import os
import time
import glob

class Capture(FloatLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root

        self.cameraObject = Camera(play=False)
        self.cameraObject.play = True
        self.cameraObject.resolution = (500, 500)
        self.cameraObject.pos_hint = {'x':0, 'y':0.2}

        anchor1 = AnchorLayout(anchor_x='left', anchor_y='bottom')
        home_button = Button(text="Home", size_hint=(0.33, .30))
        home_button.bind(on_press=self.go_home)
        anchor1.add_widget(home_button)
        self.add_widget(anchor1)

        anchor2 = AnchorLayout(anchor_x='center', anchor_y='bottom')
        cameraClick = Button(text="Take Photo", size_hint=(0.33, .30))
        cameraClick.bind(on_press=self.onCameraClick)
        anchor2.add_widget(cameraClick)
        self.add_widget(anchor2)

        anchor3 = AnchorLayout(anchor_x='right', anchor_y='bottom')
        cameraClick = Button(text="Evaluate", size_hint=(0.33, .30))
        cameraClick.bind(on_press=self.go_results)
        anchor3.add_widget(cameraClick)
        self.add_widget(anchor3)

        self.add_widget(self.cameraObject)
   
    def onCameraClick(self, *args):
        timest = time.strftime("%Y%m%d-%H%M%S")
        self.cameraObject.export_to_png(os.path.join(".\\gui\\captures", ("capture_"+timest+".png")))

    def go_home(self, instance):
        self.sn_root.go_home()

    def go_results(self, instance):
        self.sn_root.go_results()
        list_of_files = glob.glob('D:\\SolveNet\\gui\\captures\\*.png') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        self.sn_root.results_.print_results(latest_file)
