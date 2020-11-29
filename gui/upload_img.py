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

        self.home_button = Button(text="Home", size_hint=(1, .35))
        self.home_button.bind(on_press=self.go_home)
        self.select_button = Button(text="Select", size_hint=(0.5, .35))
        self.select_button.bind(on_press=self.go_results)
        self.add_widget(self.home_button)
        self.add_widget(self.select_button)

    def selected(self, filename):
        self.ids.image.source = filename[0]
        self.sel_image_path = filename[0]
    
    def go_home(self, instance):
        self.sn_root.go_home()

    def go_results(self, instance):
        self.sn_root.go_results()
        self.sn_root.results_.print_results(self.sel_image_path)

