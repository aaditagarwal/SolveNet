from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')
Config.write()

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen

from gui.home import Home
from gui.upload_img import Upload
from gui.capture_img import Capture
from gui.about_us import About

class SolveNet(App):
    def build(self):
        #from kivy.core.window import Window
        #Window.clearcolor = (1, 1, 1, 1)
        self.screen_manager = ScreenManager()

        self.home_ = Home(sn_root)
        screen = Screen(name="Home")
        screen.add_widget(self.home_)
        self.screen_manager.add_widget(screen)
        
        self.upload_img_ = Upload(sn_root)
        screen = Screen(name="Upload")
        screen.add_widget(self.upload_img_)
        self.screen_manager.add_widget(screen)

        self.capture_img_ = Capture(sn_root)
        screen = Screen(name="Capture")
        screen.add_widget(self.capture_img_)
        self.screen_manager.add_widget(screen)

        self.about_ = About(sn_root)
        screen = Screen(name="About")
        screen.add_widget(self.about_)
        self.screen_manager.add_widget(screen)

        return self.screen_manager

    def go_upload(self, *_):
        print("GOING TO UPLOAD")
        self.screen_manager.transition.direction = 'left'
        self.screen_manager.current = "Upload"
    
    def go_capture(self, *_):
        print("GOING TO CAPTURE")
        self.screen_manager.transition.direction = 'right'
        self.screen_manager.current = "Capture"

    def go_home(self, *_):
        print("GOING HOME")
        self.screen_manager.transition.direction = 'left'
        self.screen_manager.current = "Home"
    
    def go_about(self, *_):
        print("READING ABOUT THE PROJECT")
        self.screen_manager.transition.direction = 'left'
        self.screen_manager.current = "About"

if __name__ == "__main__":
    sn_root = SolveNet()
    sn_root.run()
