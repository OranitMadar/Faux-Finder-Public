import requests
import pandas as pd
import time

API_KEY = "AIzaSyCyRn5CFp9VLj2Y0HR67H6qDWcAuPcKiXg"

def search_fact_check(query, language="en"):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": API_KEY,
        "query": query,
        "languageCode": language
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {}

def build_large_dataset(queries, max_items=1000):
    data = []
    seen_texts = set()

    for query in queries:
        print(f"Searching for: {query}")
        result = search_fact_check(query)
        if not result:
            continue

        for item in result.get("claims", []):
            claim_text = item.get("text", "")
            review = item.get("claimReview", [{}])[0]
            rating = review.get("textualRating", "")
            publisher = review.get("publisher", {}).get("name", "")
            url = review.get("url", "")

            if claim_text and rating and claim_text not in seen_texts:
                label = 1 if "true" in rating.lower() else 0
                data.append({
                    "text": claim_text,
                    "label": label,
                    "rating": rating,
                    "source": publisher,
                    "url": url
                })
                seen_texts.add(claim_text)

            if len(data) >= max_items:
                break

        print(f"Current dataset size: {len(data)} items\n")
        if len(data) >= max_items:
            break

        time.sleep(1)

    return pd.DataFrame(data)

queries = [
    "Iron Swords Israel Gaza",
    "Iron Swords disinformation",
    "Israel Gaza war fake news",
    "Hamas disinformation",
    "IDF propaganda",
    "Palestinian propaganda fake",
    "October 7th fake news",
    "hospital bombing israel gaza truth",
    "UN false claims gaza",
    "truth about gaza war",
    "Israel bombing civilians fact check",
    "israel hamas news misleading",
    "massacre hoax israel",
    "hamas hostage claims fact check",
    "fake videos from gaza war",
    "fact check israel airstrike school",
    "gaza children disinformation",
    "israel war lies or truth",
    "tunnel under hospital fact check",
    "Al Jazeera disinformation Israel Gaza",
    "BBC check israel hamas",
    "snopes israel war",
    "palestinian misinformation israel",
    "gaza news real or fake",
    "confirmed news israel gaza",
    "true claims IDF",
    "verified information iron swords",
    "fact checked true israel",
    "israel gaza fact based report"

]

df = build_large_dataset(queries, max_items=1000)

# Count labels
print("Total Claims:", len(df))
print("Real News (label=1):", len(df[df["label"] == 1]))
print("Fake News (label=0):", len(df[df["label"] == 0]))

# Save
df.to_csv("iron_swords_fact_check_dataset.csv", index=False)
