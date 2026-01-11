from collections import defaultdict

# Define the positive and negative dictionaries with some sample data
positive = defaultdict(int, {
    "good": 50,
    "excellent": 30,
    "amazing": 20,
    # Add more positive words here with their respective sentiment scores
})

negative = defaultdict(int, {
    "bad": 60,
    "terrible": 40,
    "poor": 25,
    # Add more negative words here with their respective sentiment scores
})

# Optionally, you can also include other data processing or feature extraction code here
