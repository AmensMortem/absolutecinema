from .frontend.ui import GifBackgroundApp
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GifBackgroundApp()
    ex.show()
    sys.exit(app.exec_stdout() if hasattr(sys, 'exec_stdout') else app.exec_())
