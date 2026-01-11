from classifier import classify_text

def load_test_data():
    """
    Loads the test data with sentences and their actual sentiments.
    :return: List of tuples (sentence, actual_sentiment)
    """
    return [
        ("मैं खुश हूँ।", "Positive"),
        ("मुझे यह पसंद नहीं आया।", "Negative"),
        ("यह ठीक है।", "Neutral"),
        ("आज का दिन अच्छा है।", "Positive"),
        ("यह समस्या गंभीर है।", "Negative"),
    ]

def evaluate_model(test_data):
    """
    Evaluates the model on a given test dataset and calculates accuracy.
    :param test_data: List of tuples (sentence, actual_sentiment)
    :return: Accuracy percentage
    """
    correct_predictions = 0

    for text, actual_sentiment in test_data:
        predicted_sentiment, _, _ = classify_text(text)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {predicted_sentiment}, Actual Sentiment: {actual_sentiment}\n")
        if predicted_sentiment == actual_sentiment:
            correct_predictions += 1

    total_predictions = len(test_data)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def display_accuracy(accuracy):
    """
    Displays the accuracy of the model.
    :param accuracy: Accuracy percentage
    """
    print(f"Model Accuracy: {accuracy:.2f}%")

def main():
    """
    Main function to load data, evaluate the model, and display results.
    """
    print("Loading test data...")
    test_data = load_test_data()

    print("Evaluating the model...")
    accuracy = evaluate_model(test_data)

    print("Displaying results...")
    display_accuracy(accuracy)

if __name__ == "__main__":
    main()
