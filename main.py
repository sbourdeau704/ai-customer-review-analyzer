from transformers import pipeline
reviews = [
    "the bike was amazing and the battery lasted all day!",
    "Customer service was slow and I had to wait forever.",
    "The bike is okay, but I expected better quality.",
    "I love this bike and would definitely buy it again.",
    "the bike is fantastic!",
    "I wouldn't recommend this bike to anyone.", 
    "The battery isn't terrible, but I wouldn't buy it again."
]

sentiment_analyzer = pipeline("sentiment-analysis")

for review in reviews:
    result = sentiment_analyzer(review)

    print("Review:", review)
    print("AI result:", result)
    print()



