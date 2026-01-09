import pandas as pd
import re

df = pd.read_csv(r"C:\Users\seakl\Documents\I5-AMS\WR\PROJECT\data\Data Collection - Sheet1.csv")

def remove_space_between_khmer(text):
    if pd.isna(text):
        return text
    
    # Remove space ONLY if it is between Khmer characters
    text = re.sub(r'([\u1780-\u17FF])\s+([\u1780-\u17FF])', r'\1\2', text)
    
    # Optional: normalize remaining multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

for col in df.columns:
    df[col] = df[col].apply(remove_space_between_khmer)

df.to_csv("data_cleaned_all.csv", index=False, encoding="utf-8")

print("✅ All Khmer inter-word spaces removed dynamically")
