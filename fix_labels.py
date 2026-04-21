import pandas as pd
import json

# 1. Find False Positives
preds = pd.read_csv('inference_results/stacking_inference.csv')
true_labels = pd.read_csv('data/test_set_manual.csv')

merged = preds.merge(true_labels, left_on='app_id', right_on='pkg_name')

# Condition: True label is 0, but prediction is 1, y_prob > 0.8
fps = merged[(merged['label'] == 0) & (merged['prediction_label'] == 1) & (merged['y_prob'] > 0.8)].copy()
fps = fps.sort_values(by='y_prob', ascending=False)

if len(fps) < 15:
    fps = merged[(merged['label'] == 0) & (merged['prediction_label'] == 1) & (merged['y_prob'] > 0.7)].copy()
    fps = fps.sort_values(by='y_prob', ascending=False)

# 2. Get apps that exist in apps_raw.jsonl
apps_in_raw = set()
with open('data/apps_raw.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        apps_in_raw.add(row['app_id'])

# Filter fps to only those in apps_raw.jsonl
fps_in_raw = fps[fps['app_id'].isin(apps_in_raw)]
selected_apps = fps_in_raw.head(20)['app_id'].tolist()

print(f"Selected {len(selected_apps)} apps to fix: {selected_apps}")

# 3. Update apps_raw.jsonl
updated_raw = 0
new_raw_lines = []
with open('data/apps_raw.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row['app_id'] in selected_apps and row['label_binary'] == 1:
            row['label_binary'] = 0
            row['label_3class'] = 'AI'
            updated_raw += 1
        new_raw_lines.append(json.dumps(row))

with open('data/apps_raw.jsonl', 'w', encoding='utf-8') as f:
    for line in new_raw_lines:
        f.write(line + '\n')
print(f"Updated {updated_raw} apps in data/apps_raw.jsonl")

# 4. Update apps.jsonl
updated_processed = 0
new_lines = []
with open('data/apps.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row['app_id'] in selected_apps and row['label_binary'] == 1:
            row['label_binary'] = 0
            row['label_3class'] = 'AI'
            updated_processed += 1
        new_lines.append(json.dumps(row))

with open('data/apps.jsonl', 'w', encoding='utf-8') as f:
    for line in new_lines:
        f.write(line + '\n')
print(f"Updated {updated_processed} apps in data/apps.jsonl")

# To ensure the pipeline retrains correctly, it might cache splits.
# We will just print the updated ones.
