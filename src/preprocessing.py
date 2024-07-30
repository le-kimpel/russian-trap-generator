import pandas as pd
import re

df = pd.read_csv("../data/archive/genius_ru_songs_70.csv")

lyric_col = df["lyrics"]

'''
Naive way: stuff all lyrics into one blob.
'''

with open("out.txt", "w", encoding="utf-8") as f:
    for lyric in lyric_col:
        pattern = re.compile(r"[^\w\n']+", re.UNICODE)
        lyric_mod = re.sub(pattern, ' ', lyric)
        lyric_mod = lyric_mod.replace('\n', ' ')
        lyric_mod = lyric_mod.replace('\u2005', ' ')
        lyric_mod = lyric_mod.replace('\u205f', ' ')
        f.write(lyric_mod)
        

