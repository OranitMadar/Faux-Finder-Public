import pandas as pd

# טוענים את הקובץ
df = pd.read_csv("zenodo_cut_for_balance.csv")

# מסירים כפילויות לפי כותרת
df = df.drop_duplicates(subset='headlines')

# מפרידים את השורות לפי תווית
df_true = df[df['outcome'] == 1]
df_fake = df[df['outcome'] == 0]

# איזון - דוגמים את כמות השקר לפי מספר האמת
df_fake_sampled = df_fake.sample(n=len(df_true), random_state=42)

# מאחדים ומערבבים
balanced_df = pd.concat([df_true, df_fake_sampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# בדיקה
print(balanced_df['outcome'].value_counts())

# שמירה לקובץ חדש
balanced_df.to_csv("balanced_dataset.csv", index=False)
counts = balanced_df['outcome'].value_counts()
print(f"אמת (1): {counts.get(1, 0)} שורות")
print(f"שקר (0): {counts.get(0, 0)} שורות")
# סופרת שורות שמופיעות יותר מפעם אחת בכל העמודות
duplicate_rows = balanced_df.duplicated()
print(f"מספר שורות כפולות (בדיוק אותו תוכן בכל העמודות): {duplicate_rows.sum()}")


