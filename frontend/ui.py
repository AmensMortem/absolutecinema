import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt


class GifBackgroundApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        y,x = 889, 500
        self.setFixedSize(y, x)
        self.setWindowTitle("Main")

        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, y, x)

        self.movie = QMovie("sad-koi.gif")
        self.bg_label.setMovie(self.movie)
        self.bg_label.setScaledContents(True)  # Ensures it fits the label exactly
        self.movie.start()

        # 3. Create the UI widgets
        self.input_field = QLineEdit()
        self.input_field.setFixedSize(y, 50)

        self.input_field.setPlaceholderText("Description")

        self.submit_btn = QPushButton("Submit")
        # Connect the button click to our processing function
        self.submit_btn.clicked.connect(self.handle_submit)

        self.result_label = QLabel("Result will appear here")

        # Styling the text so it's readable on top of a background
        self.result_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)

        # 4. Use a layout to stack widgets on top of the background
        layout = QVBoxLayout()
        layout.addStretch(1)  # Pushes widgets toward the center/bottom if desired
        layout.addWidget(self.input_field)
        layout.addWidget(self.submit_btn)
        layout.addWidget(self.result_label)
        layout.addStretch(1)

        self.setLayout(layout)

    def backend(self, user_text):
        if not user_text.strip():
            return "You didn't type anything!"
        return f"Processed: {user_text.upper()}"

    def handle_submit(self):
        entered_text = self.input_field.text()
        output = self.backend(entered_text)
        self.result_label.setText(output)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GifBackgroundApp()
    ex.show()
    sys.exit(app.exec_stdout() if hasattr(sys, 'exec_stdout') else app.exec_())