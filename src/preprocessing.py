import pandas as pd

df = pd.read_csv("../data/archive/genius_ru_songs_70.csv")

lyric_col = df["lyrics"]

'''
Naive way: stuff all lyrics into one blob.
'''
with open("out.txt", "w") as f:
    for lyric in lyric_col:
        f.write(lyric)
