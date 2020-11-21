from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label

class About(FloatLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root

        self.about_text1 = Label(text="[size=25][b]SolveNet[/b][/size]\nMade by Aadit Agarwal, Aastha Jain and Himanshu Ruhela", markup=True, halign="center", valign="middle")

        self.home_button = Button(text="Home")
        self.home_button.bind(on_press=self.go_home)
        self.home_button.pos_hint = {'x':0.4, 'y':0.1}
        self.home_button.size_hint = (0.2,0.05)

        self.add_widget(self.home_button)
        self.add_widget(self.about_text1)
    
    def go_home(self, instance):
        self.sn_root.go_home()
