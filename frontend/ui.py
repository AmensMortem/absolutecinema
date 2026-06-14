from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton
from PyQt5.QtGui import QMovie, QFont
from PyQt5.QtCore import Qt
from absolutecinema.backend import interface

model, tokenizer, mlb, thresholds, max_len = interface.load_model("./saved_model")


class GifBackgroundApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        y, x = 889, 500
        self.setFixedSize(y, x)
        self.setWindowTitle("Main")

        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, y, x)

        self.movie = QMovie("sad-koi.gif")
        self.bg_label.setMovie(self.movie)
        self.bg_label.setScaledContents(True)
        self.movie.start()

        self.input_field = QLineEdit()
        self.input_field.setFixedSize(y, 50)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;  /* Dark gray background */
                color: #ffffff;              /* White text color */
                border: 2px solid #555555;   /* Dark border */
                border-radius: 5px;          /* Slightly rounded corners */
                padding: 5px;
            }
        """)

        self.input_field.setPlaceholderText("Description")

        self.button = QPushButton("Submit")
        self.button.setFixedSize(150, 50)
        self.button.setFont(QFont("Montserrat Alternates SemiBold", 10))
        self.button.clicked.connect(self.handle_submit)

        self.button.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;  /* Green background */
                color: white;                /* White text */
                border-radius: 4px;
                font-weight: bold;

            }
            QPushButton:hover {
                background-color: #45a049;  /* Darker green when you hover over it */
            }
            QPushButton:pressed {
                background-color: #367c39;  /* Even darker green when clicked */
            }
        """)
        self.result_label = QLabel("Result will appear here")
        self.result_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.input_field)
        layout.addWidget(self.button, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_label)
        layout.addStretch(1)

        self.setLayout(layout)

    def backend(self, user_text):
        if not user_text.strip():
            return "You didn't type anything!"
        genres = ", ".join(interface.predict(
            text=user_text,
            model=model,
            tokenizer=tokenizer,
            mlb=mlb,
            thresholds=thresholds,
            max_len=max_len,
            mode="precision"
        ))
        return genres

    def handle_submit(self):
        entered_text = self.input_field.text()
        output = self.backend(entered_text)
        self.result_label.setText(output)
