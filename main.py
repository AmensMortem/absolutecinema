from PyQt5.QtWidgets import QApplication
import sys
from backend import interface, final_version, save_model
from frontend import ui

if __name__ == '__main__':
    print("App is running, wait for some time...")

    app = QApplication(sys.argv)
    ex = ui.GifBackgroundApp()
    ex.show()
    sys.exit(app.exit() if hasattr(sys, 'exec_stdout') else app.exec_())
