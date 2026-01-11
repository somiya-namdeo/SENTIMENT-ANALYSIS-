# classifier.py
from lexicon import analyze_sentiment
from translator import translate_to_hindi


def is_english(text):
    """
    Detect if text is likely English based on simple heuristics.
    :param text: Input string
    :return: Boolean
    """
    # Simple heuristic: If >50% ASCII letters, classify as English
    english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    return english_chars > len(text) / 2


def classify_text(text):
    """
    Analyzes sentiment and translates English to Hindi if necessary.
    :param text: Input string
    :return: Sentiment, polarity, and translated text if English
    """
    if is_english(text):
        translated_text = translate_to_hindi(text)  # Translate English text
    else:
        translated_text = text  # Keep the original if it's Hindi

    sentiment, polarity = analyze_sentiment(translated_text)
    return sentiment, polarity, translated_text
