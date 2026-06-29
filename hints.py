import pandas as pd

df = pd.read_csv('class6_to_12_with_hints.csv')

# सिर्फ Maths questions लो
df_maths = df[df['context'].str.contains('Mathematics', case=False, na=False)]

print(f"Total Maths questions (Class 6-12): {len(df_maths)}")

# नई फाइल सेव करो
df_maths.to_csv('class6_to_12_maths_with_hints.csv', index=False)

print("✅ Done! File saved as 'class6_to_12_maths_with_hints.csv'")
print("अब ये सिर्फ Maths के questions हैं")