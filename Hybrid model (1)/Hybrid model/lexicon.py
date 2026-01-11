# lexicon.py

# Sentiment Lexicon
lexicon = {
    'खुश': 1,
    'शुभ': 1,
    'आशा': 1,
    'संतुष्ट': 1,
    'सफलता': 1,
    'पसंद': 1,
    'अच्छा': 1,
    'बुरा': -1,
    'दुःख': -1,
    'खराब': -1,
    'निराश': -1,
    'विरोध': -1,
    'घबराना': -1,
    'संकट': -1,
    'नफरत': -1,
    'शिकायत': -1,
    'बेरोज़गारी': -1,
    'संगीन': -1,
    'संकोच': -1,
    'मांग': 0,
    'बातचीत': 0,
    'समझ': 0,
}

def analyze_sentiment(text):
    """
    Analyzes Hindi text for sentiment and calculates polarity.
    :param text: Input string
    :return: Sentiment and polarity
    """
    words = text.split()
    score = 0
    for word in words:
        if word in lexicon:
            score += lexicon[word]

    # Determine sentiment
    if score > 0:
        sentiment = "Positive"
    elif score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Polarity is the score itself
    polarity = score

    return sentiment, polarity
