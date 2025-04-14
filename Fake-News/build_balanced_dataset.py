
import requests
import pandas as pd
import time
import newspaper
import matplotlib.pyplot as plt

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

def collect_fake_news(queries, target_count=500):
    data = []
    seen_texts = set()

    for query in queries:
        print(f"Searching for fake news: {query}")
        result = search_fact_check(query)
        if not result:
            continue

        for item in result.get("claims", []):
            if len(data) >= target_count:
                break

            claim_text = item.get("text", "")
            review = item.get("claimReview", [{}])[0]
            rating = review.get("textualRating", "")
            publisher = review.get("publisher", {}).get("name", "")
            url = review.get("url", "")

            if claim_text and rating and claim_text not in seen_texts:
                label = 0
                data.append({
                    "text": claim_text,
                    "label": label,
                    "rating": rating,
                    "source": publisher,
                    "url": url
                })
                seen_texts.add(claim_text)

        print(f"→ Fake count so far: {len(data)}\n")
        if len(data) >= target_count:
            break

        time.sleep(1)

    return pd.DataFrame(data)

def collect_real_news(sources, target_count=500):
    data = []
    seen_titles = set()

    for source in sources:
        print(f"Scraping real news from: {source}")
        paper = newspaper.build(source, memoize_articles=False)

        for article in paper.articles:
            if len(data) >= target_count:
                break
            try:
                article.download()
                article.parse()
                article.nlp()

                title = article.title.strip()
                if title and title not in seen_titles and len(title.split()) > 5:
                    data.append({
                        "text": title,
                        "label": 1,
                        "source": source,
                        "url": article.url
                    })
                    seen_titles.add(title)
            except:
                continue

        print(f"→ Real count so far: {len(data)}\n")
        if len(data) >= target_count:
            break

    return pd.DataFrame(data)

fake_queries = [
    "fake hostage video gaza",
    "misleading idf statement",
    "false photo israel bombing",
    "gaza deaths fake report",
    "AI fake report gaza",
    "unverified video hamas",
    "fake statistics gaza war",
    "misleading map israel attack",
    "fabricated timeline october 7",
    "false bombing claim gaza school",
    "twitter false narrative gaza",
    "unconfirmed claim israel genocide",
    "disproved claim UN gaza",
    "false viral claim israel soldier",
    "gaza hospital bombing untrue",
    "false social media report gaza",
    "tiktok fake massacre video",
    "hamas manipulated video gaza",
    "gaza child death fake news",
    "false idf bombing al jazeera",
    "Iron Swords Israel Gaza fake news",
    "October 7 hoax",
    "Israel bombing lies",
    "Hamas disinformation",
    "Palestinian fake claims",
    "Israel propaganda false",
    "IDF hospital attack fake",
    "Israel war misleading news",
    "False reports Gaza war",
    "Fake videos Iron Swords",
    "False IDF videos",
    "hoax Israel Hamas war",
    "fake headlines gaza israel",
    "gaza war conspiracy",
    "misleading news israel",
    "Iron Swords fake news",
    "Gaza war disinformation",
    "IDF lies Gaza",
    "False Hamas claims",
    "Fake news about Israel",
    "October 7th hoax",
    "fabricated reports gaza",
    "hamas fake massacre",
    "false IDF war crimes",
    "israel gaza bombing fake video",
    "Fake video hospital Gaza",
    "Doctored video israel airstrike",
    "falsified footage IDF",
    "hamas staged funeral",
    "fake hostages hamas video",
    "False narrative UN Gaza",
    "TikTok fake videos Gaza",
    "Twitter Israel war hoax",
    "Viral fake news israel gaza",
    "Telegram disinformation Gaza",
    "Facebook manipulated gaza news",
    "AI generated footage gaza war",
    "Deepfake hostage israel",
    "Edited missile footage IDF",
    "fabricated explosion israel gaza",
    "Manipulated gaza images",
    "Iran propaganda israel war",
    "Russia fake news Gaza",
    "Al Jazeera fake IDF reports",
    "UN report manipulated israel",
    "Snopes israel gaza hoax",
    "FactCheck.org hamas false",
    "PolitiFact gaza conflict"

]

real_news_sources = [
    "https://www.bbc.com/news/world-middle-east",
    "https://www.reuters.com/world/middle-east/",
    "https://apnews.com/hub/israel-hamas-war",
    "https://www.nytimes.com/section/world/middleeast",
    "https://www.timesofisrael.com/",
    "https://www.haaretz.com/middle-east-news",
    "https://edition.cnn.com/middleeast",
    "https://www.aljazeera.com/news/middleeast",
    "https://www.jpost.com/breaking-news",
    "https://www.washingtonpost.com/world/middle-east"
]

df_fake = collect_fake_news(fake_queries, target_count=500)
df_real = collect_real_news(real_news_sources, target_count=500)

df_combined = pd.concat([df_fake, df_real], ignore_index=True)
df_combined.to_csv("iron_swords_balanced_dataset_expanded.csv", index=False)

label_counts = df_combined["label"].value_counts()
plt.bar(label_counts.index.map({0: "Fake News", 1: "Real News"}), label_counts.values, color=["red", "green"])
plt.title("Fake vs Real News Distribution")
plt.ylabel("Number of Samples")
plt.show()
