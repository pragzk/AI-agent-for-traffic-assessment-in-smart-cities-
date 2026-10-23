import pandas as pd
import matplotlib.pyplot as plt
import os

cols = [
    'Severity', 'Start_Time', 'Weather_Condition',
    'Sunrise_Sunset', 'Temperature(F)',
    'Visibility(mi)', 'Wind_Speed(mph)'
]

print("Loading... (may take 30 seconds)")
df = pd.read_csv('US_Accidents_March23.csv', usecols=cols, nrows=200000)
print(f"Loaded: {df.shape}")

df.to_csv('accidents_small.csv', index=False)
print("Saved accidents_small.csv")

df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df.dropna(inplace=True)
print(f"After cleaning: {df.shape}")



#Print the numbers needed
print("\nPEAK HOURS (top 5):")
print(df['Hour'].value_counts().head(5).sort_index())

print("\nAVG SEVERITY BY WEATHER (top 10):")
print(df.groupby('Weather_Condition')['Severity']
        .mean().sort_values(ascending=False).head(10).round(2))

print("\nDAY vs NIGHT SEVERITY:")
print(df.groupby('Sunrise_Sunset')['Severity'].mean().round(3))

print("\nVISIBILITY STATS:")
print(df['Visibility(mi)'].describe().round(2))

print("\nSEVERITY DISTRIBUTION:")
print(df['Severity'].value_counts().sort_index())


plt.figure(figsize=(12, 4))
df['Hour'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig('plot_hour.png'); plt.show()

#Plot:Severity day vs night
plt.figure(figsize=(6, 4))
df.groupby('Sunrise_Sunset')['Severity'].mean().plot(
    kind='bar', color=['gold', 'navy'])
plt.title('Avg Severity — Day vs Night')
plt.tight_layout(); plt.savefig('plot_daynight.png'); plt.show()

#Plot: top weather conditions
plt.figure(figsize=(12, 4))
df['Weather_Condition'].value_counts().head(10).plot(
    kind='bar', color='coral')
plt.title('Top 10 Weather Conditions in Accidents')
plt.tight_layout(); plt.savefig('plot_weather.png'); plt.show()