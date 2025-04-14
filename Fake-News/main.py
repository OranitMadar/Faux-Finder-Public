import requests
import pandas as pd
from time import sleep

API_KEY = "AIzaSyCyRn5CFp9VLj2Y0HR67H6qDWcAuPcKiXg"  # שימי פה את המפתח שלך
QUERY = "iron swords OR חרבות ברזל"
MAX_RESULTS = 1000  # נמשוך הרבה כדי לסנן
LANG = "en"

def get_claims_from_google(query, language="en", page_size=100):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "languageCode": language,
        "pageSize": page_size,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("claims", [])
    else:
        print("❌ Error:", response.status_code)
        return []

def extract_claim_data(claims):
    real = []
    fake = []
    for c in claims:
        text = c.get("text", "")
        claimant = c.get("claimant", "")
        reviews = c.get("claimReview", [])
        for r in reviews:
            rating = r.get("textualRating", "").lower()
            source = r.get("publisher", {}).get("name", "")
            url = r.get("url", "")

            if "false" in rating or "fake" in rating or "incorrect" in rating or "pants on fire" in rating:
                fake.append({"text": text, "label": 0, "source": source, "url": url})
            elif "true" in rating or "correct" in rating or "accurate" in rating:
                real.append({"text": text, "label": 1, "source": source, "url": url})
    return real, fake

# שליפת טענות
print("🔍 מחפש טענות...")
claims = get_claims_from_google(QUERY, LANG)

print(f"🔢 נמצאו {len(claims)} טענות. מפצל לפי אמינות...")
real_claims, fake_claims = extract_claim_data(claims)

# סינון ל-500 מכל סוג
real_claims = real_claims[:500]
fake_claims = fake_claims[:500]

# חיבור לדאטהסט אחד
dataset = real_claims + fake_claims
df = pd.DataFrame(dataset)

# ערבוב
df = df.sample(frac=1).reset_index(drop=True)

# שמירה
df.to_csv("iron_swords_dataset.csv", index=False)
print("✅ דאטהסט נשמר לקובץ: iron_swords_dataset.csv")
