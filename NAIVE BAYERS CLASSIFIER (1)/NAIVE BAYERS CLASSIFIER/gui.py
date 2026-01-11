from PyQt5.QtWidgets import QMainWindow, QApplication, QPlainTextEdit, QLineEdit, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont
import sys
from classifier import classify_sentiment, evaluate_classifier

class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.title = 'Sentiment Analysis of Hindi Text'
        self.left = 20
        self.top = 20
        self.width = 500
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create a central widget to hold the layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create the layout
        layout = QVBoxLayout()

        # Set font for Hindi text
        hindi_font = QFont("Mangal", 14)

        # Create textbox for displaying result (output box)
        self.DisplayTextbox = QPlainTextEdit(self)
        self.DisplayTextbox.setReadOnly(True)
        self.DisplayTextbox.setFixedHeight(100)
        self.DisplayTextbox.setFont(hindi_font)
        self.DisplayTextbox.setStyleSheet("""
            background-color: #f0f8ff;
            color: #333333;
            border: 2px solid #8a2be2;
            border-radius: 10px;
            padding: 10px;
        """)
        layout.addWidget(self.DisplayTextbox)

        # Create textbox for entering input text (input box)
        self.EnterTextbox = QLineEdit(self)
        self.EnterTextbox.setFixedHeight(150)
        self.EnterTextbox.setFixedWidth(400)
        self.EnterTextbox.setFont(hindi_font)
        self.EnterTextbox.setStyleSheet("""
            background-color: #fff8dc;
            color: #333333;
            border: 2px solid #ff6347;
            border-radius: 10px;
            padding: 10px;
        """)
        layout.addWidget(self.EnterTextbox)

        # Add a button for file selection
        self.selectFileButton = QLineEdit(self)
        self.selectFileButton.setPlaceholderText("Select a test file for accuracy testing")
        self.selectFileButton.setFixedHeight(30)
        self.selectFileButton.setFont(hindi_font)
        self.selectFileButton.setStyleSheet("""
            background-color: #dcdcdc;
            color: #333333;
            border: 2px solid #4682b4;
            border-radius: 5px;
            padding: 5px;
        """)
        layout.addWidget(self.selectFileButton)

        # Set the layout to the central widget
        central_widget.setLayout(layout)

        # Connect enter textbox to function on_click
        self.EnterTextbox.returnPressed.connect(self.on_click)

        # Connect file selection to the function
        self.selectFileButton.mousePressEvent = self.select_file

        self.show()

    @pyqtSlot()
    def on_click(self):
        textfromgui = self.EnterTextbox.text()

        # Debug print to check if text is being received
        print(f"Received text for classification: {textfromgui}")

        # Classify the sentiment of the input text
        sentiment = classify_sentiment(textfromgui)
        
        # Debug print to check sentiment output
        print(f"Classified sentiment: {sentiment}")
        
        if sentiment == "positive":
            self.DisplayTextbox.setPlainText("सकारात्मक भावना (POSITIVE SENTIMENT)")
        elif sentiment == "negative":
            self.DisplayTextbox.setPlainText("नकारात्मक भावना (NEGATIVE SENTIMENT)")
        else:
            self.DisplayTextbox.setPlainText("तटस्थ भावना (NEUTRAL SENTIMENT)")

    def select_file(self, event):
        # Open a file dialog to select a test file
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Sentiment Test Data File", "", "Text Files (*.txt)")

        if file_name:
            self.selectFileButton.setText(file_name)  # Show file name in the textbox
            self.calculate_accuracy(file_name)

    def calculate_accuracy(self, file_name):
        # Debug print to check if file name is being received
        print(f"Selected file for accuracy testing: {file_name}")

        accuracy, cm = evaluate_classifier(file_name)
        
        # Debug print to check accuracy and confusion matrix
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Display accuracy
        cm_str = '\n'.join(['\t'.join(map(str, row)) for row in cm])
        self.DisplayTextbox.setPlainText(f"Accuracy: {accuracy:.2f}%\n\nConfusion Matrix:\n{cm_str}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
