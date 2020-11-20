from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout

Builder.load_string("""
<Upload>:
    id: upload_widget
    FileChooserListView:
        id: filechooser
        on_selection: upload_widget.selected(filechooser.selection)
    Image:
        id: image
        source: ""
""")

class Upload(GridLayout):
    def __init__(self, sn_root, **kwargs):
        super().__init__(**kwargs)
        self.sn_root = sn_root
        self.cols = 2

        self.home_button = Button(text="Home")
        self.home_button.bind(on_press=self.go_home)
        self.select_button = Button(text="Select")
        self.select_button.bind(on_press=self.select_path)
        self.add_widget(self.home_button)
        self.add_widget(self.select_button)

    def selected(self, filename):
        self.ids.image.source = filename[0]
        global image_path
        image_path = filename[0]
    
    def go_home(self, instance):
        self.sn_root.go_home()

    def select_path(self, filename):
        return image_path

