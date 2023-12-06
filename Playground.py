import pandas as pd
import random

df = pd.read_csv('new_train.csv')
d = {}

for index, row in df.iterrows():
    label = row['label']
    image_id = row['image_id']
    if label not in d:
        d[label] = []
    d[label].append(image_id)

data = []
for label, image_ids in d.items():
    random.shuffle(image_ids)
    image_ids = image_ids[:7200]
    for image_id in image_ids:
        data.append((image_id, label))

# Create a DataFrame from the list
df = pd.DataFrame(data, columns=['image_id', 'label'])

# Save the DataFrame to a new CSV file
df.to_csv('averaged_train.csv', index=False)

print(len(data))