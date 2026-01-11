from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt
from classifier import classify_text
import sys


class SentimentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hindi Sentiment Analysis")
        self.setGeometry(100, 100, 600, 400)

        # Set up the main UI
        self.setup_ui()

    def setup_ui(self):
        # Main layout
        self.layout = QVBoxLayout()

        # Adding styles
        self.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                color: #FFFFFF;
            }
            QLineEdit {
                font-size: 14px;
                padding: 5px;
                border: 2px solid #5A9;
                border-radius: 5px;
                background-color: #E6F4F1;
            }
            QPushButton {
                font-size: 14px;
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            """
        )

        # Set background color
        self.set_background_color("#2D3E50")

        # Adding widgets to the layout
        self.label = QLabel("Enter English or Hindi text for sentiment analysis:")
        self.label.setFont(QFont("Arial", 14))
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type here...")
        self.layout.addWidget(self.input_box)

        self.button = QPushButton("Analyze Sentiment")
        self.button.clicked.connect(self.analyze_sentiment)
        self.layout.addWidget(self.button)

        # Result labels
        self.result_label = QLabel("Sentiment: ")
        self.result_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.result_label)

        self.polarity_label = QLabel("Polarity Score: ")
        self.polarity_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.polarity_label)

        self.translation_label = QLabel("")
        self.translation_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.translation_label)

        # Set up the container widget and layout
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def set_background_color(self, color):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)

    def analyze_sentiment(self):
        user_input = self.input_box.text()
        sentiment, polarity, translated_text = classify_text(user_input)

        # Update GUI with results
        self.result_label.setText(f"Sentiment: {sentiment}")
        self.polarity_label.setText(f"Polarity Score: {polarity}")

        # Show translation if the input was English
        if translated_text != user_input:
            self.translation_label.setText(f"Hindi Translation: {translated_text}")
        else:
            self.translation_label.setText("")  # Clear if input is already Hindi


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())
