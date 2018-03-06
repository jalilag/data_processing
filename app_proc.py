from user_idas import DasData
from user_qt2 import UQapp,UQaction
from lang import txt
from app_style import sty
class Proc():
	application = None
	win1 = None	
	lang = "fr"
	def __init__(self,application):
		self.application = application
		self.win1 = UQapp(title="Test")
		self.win1.style = "../media/css/"
		self.menubar()
		self.win1.center(1)
		self.win1.showMaximized()

	def menubar(self):
		# Quit
		k = self.get_args(name_id="ACTION_EXIT",parent=self.win1)
		# exitAction = UQaction(self.get_args(name_id="ACTION_EXIT",parent=self.win1))
		# print(type(k),k)
		exitAction = UQaction(**k)
		# exitAction.triggered.connect(self.sig_close_app)
		# Config
		configAction = UQaction(**self.get_args(name_id="ACTION_CONFIG",parent=self.win1))
		# configAction.triggered.connect(partial(self.sig_load_view,0,0))
		# # Home
		homeAction = UQaction(**self.get_args(name_id="HOME_ACTION",parent=self.win1))
		# homeAction.triggered.connect(partial(self.sig_load_view,1,1))
		# Menu
		menubar = self.win1.menuBar()
		fileMenu = menubar.addMenu('File')
		fileMenu.addAction(homeAction)
		fileMenu.addAction(configAction)
		fileMenu.addAction(exitAction)


	def get_args(self,**kwargs):
		if "name_id" not in kwargs: return kwargs
		if kwargs["name_id"] not in sty: return kwargs 
		kwargs.update(sty[kwargs["name_id"]])  
		if "title" in kwargs:
			kwargs["title"] = txt[kwargs["title"]][self.lang]
		return kwargs
