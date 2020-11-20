from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen

class Home(GridLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root
        self.cols = 2

        self.upload_option_button = Button(text="Upload")
        self.upload_option_button.bind(on_press=self.choose_upload)

        self.capture_option_button = Button(text="Capture")
        self.capture_option_button.bind(on_press=self.choose_capture)

        self.add_widget(self.upload_option_button)
        self.add_widget(self.capture_option_button)
    
    def choose_upload(self, instance):
        self.sn_root.go_upload()
    def choose_capture(self, instance):
        self.sn_root.go_capture()
