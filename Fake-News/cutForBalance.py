import pandas as pd
#
# # טוענים את הקובץ
# df = pd.read_csv("covidSelfDataset.csv")
#
# initial_count = len(df)
#
# # מסירים כפילויות לפי כותרת
# df = df.drop_duplicates(subset='text')
#
# # סופרים את מספר השורות לאחר הסרת כפילויות
# final_count = len(df)
#
# # מחשבים את מספר השורות שנמחקו
# duplicates_removed = initial_count - final_count
#
# # מדפיסים את מספר השורות שנמחקו
# print(f"מספר שורות כפולות שנמחקו: {duplicates_removed}")
# # מפרידים את השורות לפי תווית
# df_true = df[df['outcome'] == 'real']
# df_fake = df[df['outcome'] == 'fake']
#
# # איזון - דוגמים את כמות השקר לפי מספר האמת
# df_fake_sampled = df_fake.sample(n=len(df_true), random_state=42)
#
# # מאחדים ומערבבים
# balanced_df = pd.concat([df_true, df_fake_sampled])
# balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # בדיקה
# print(balanced_df['outcome'].value_counts())
#
# # שמירה לקובץ חדש
# balanced_df.to_csv("covidSelfDatasetBalanced.csv", index=False)
# counts = balanced_df['outcome'].value_counts()
# print(f"אמת (real): {counts.get('real', 0)} שורות")
# print(f"שקר (fake): {counts.get('fake', 0)} שורות")
#
# # סופרת שורות שמופיעות יותר מפעם אחת בכל העמודות
# duplicate_rows = balanced_df.duplicated()
# print(f"מספר שורות כפולות (בדיוק אותו תוכן בכל העמודות): {duplicate_rows.sum()}")
#
#
#import pandas as pd

# --- שלב 1: טען את הקבצים ---
df_self = pd.read_csv("covidSelfDataset.csv")           # כולל outcome: 'real' / 'fake'
df_zenodo = pd.read_csv("zenodo_original.csv")          # כולל outcome: 0 / 1 ועמודת headlines

# --- שלב 2: אחידות שמות וערכים ---
df_zenodo = df_zenodo.rename(columns={'headlines': 'text'})          # התאמת שם עמודה
df_zenodo['outcome'] = df_zenodo['outcome'].replace({0: 'fake', 1: 'real'})  # המרה לערכים אחידים

# --- שלב 3: איחוד והסרת כפילויות ---
combined_df = pd.concat([
    df_self[['text', 'outcome']],
    df_zenodo[['text', 'outcome']]
], ignore_index=True)

combined_df = combined_df.drop_duplicates(subset='text')

# --- שלב 4: חלוקה ואיזון ---
df_true = combined_df[combined_df['outcome'] == 'real']
df_fake = combined_df[combined_df['outcome'] == 'fake']

# איזון לפי כמות אמת
df_fake_sampled = df_fake.sample(n=len(df_true), random_state=42)

# --- שלב 5: איחוד וערבוב ---
balanced_df = pd.concat([df_true, df_fake_sampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- שלב 6: שמירת קבצים ---
df_true.to_csv("all_real_from_both_sources.csv", index=False)
df_fake_sampled.to_csv("sampled_fake_to_match_real.csv", index=False)
balanced_df.to_csv("balanced_combined_from_self_and_zenodo.csv", index=False)
# comment
print("✅ נוצרו הקבצים:")
print("• all_real_from_both_sources.csv")
print("• sampled_fake_to_match_real.csv")
print("• balanced_combined_from_self_and_zenodo.csv")
