from classes import *
import constants as c
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

metadata = Metadata(c.FULL_DATA_DIR, sample=False)
df = metadata.balanced()

# Iterate over all ids and extract features
for _, row in df.iterrows():
    id, folder, _ = row
    try:
        Video(metadata, id, 50, False, "")._export_features('./data/landmarks/')
    except Exception as e:
        print(f"Error with {id}: {e}")
        continue
