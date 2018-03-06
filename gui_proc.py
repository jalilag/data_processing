import sys
from PyQt5.QtWidgets import QApplication
from app_proc import Proc

application = QApplication(sys.argv)
appBuild = Proc(application)
application.exec_()