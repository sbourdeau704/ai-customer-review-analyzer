from transformers import pipeline
from collections import Counter

# Sample customer reviews
reviews = [
    "The bike was amazing and the battery lasted all day!",
    "Customer service was slow and I had to wait forever.",
    "The bike is okay, but I expected better quality.",
    "I love this bike and would definitely buy it again.",
    "The bike is fantastic!",
    "I wouldn't recommend this bike to anyone.",
    "The battery isn't terrible, but I wouldn't buy it again."
]

# Load AI sentiment model
sentiment_analyzer = pipeline("sentiment-analysis")

# Simple themes we want to look for
themes = {
    "battery": ["battery", "charge", "charging"],
    "customer service": ["customer service", "service", "support"],
    "quality": ["quality", "build", "durable"],
    "recommendation": ["recommend", "buy", "purchase"]
}

sentiment_counts = Counter()
theme_counts = Counter()

print("\nAI CUSTOMER REVIEW ANALYZER")
print("=" * 40)

for review in reviews:
    result = sentiment_analyzer(review)[0]

    label = result["label"]
    confidence = result["score"]

    sentiment_counts[label] += 1

    review_lower = review.lower()

    detected_themes = []

    for theme, keywords in themes.items():
        if any(keyword in review_lower for keyword in keywords):
            theme_counts[theme] += 1
            detected_themes.append(theme)

    print(f"\nReview: {review}")
    print(f"Sentiment: {label}")
    print(f"Confidence: {confidence:.2%}")

    if detected_themes:
        print("Themes:", ", ".join(detected_themes))
    else:
        print("Themes: none detected")

print("\n" + "=" * 40)
print("SUMMARY")
print("=" * 40)

print("\nSentiment totals:")
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count}")

print("\nMost discussed themes:")
for theme, count in theme_counts.most_common():
    print(f"{theme}: {count}")
