from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen

class Home(FloatLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root
        self.cols = 2

        self.upload_option_button = Button(text="Upload")
        self.upload_option_button.bind(on_press=self.choose_upload)

        self.capture_option_button = Button(text="Capture")
        self.capture_option_button.bind(on_press=self.choose_capture)

        self.about_button = Button(text="About")
        self.about_button.bind(on_press=self.read_about)

        self.upload_option_button.size_hint = (0.25,0.25)
        self.upload_option_button.pos_hint = {'x':0.15, 'y':0.6}

        self.capture_option_button.size_hint = (0.25,0.25)
        self.capture_option_button.pos_hint = {'x':0.6, 'y':0.6}

        self.about_button.size_hint = (0.25,0.1)
        self.about_button.pos_hint = {'x':0.375, 'y':0.3}

        self.add_widget(self.upload_option_button)
        self.add_widget(self.capture_option_button)
        self.add_widget(self.about_button)
    
    def choose_upload(self, instance):
        self.sn_root.go_upload()
        
    def choose_capture(self, instance):
        self.sn_root.go_capture()

    def read_about(self, instance):
        self.sn_root.go_about()
