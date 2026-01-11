# translator.py
from googletrans import Translator


def translate_to_hindi(text):
    """
    Translates English text to Hindi
    :param text: Input string
    :return: Translated text in Hindi
    """
    translator = Translator()
    try:
        translated_text = translator.translate(text, dest='hi').text
        return translated_text
    except Exception as e:
        print(f"Error in translation: {e}")
        return text  # Return the original text if translation fails
