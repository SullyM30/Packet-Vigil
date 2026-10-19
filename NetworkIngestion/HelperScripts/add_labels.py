"""
Quick script to add attack/benign labels to the replay CSV.
"""
import csv

INPUT_CSV = "NetworkIngestion/SSDP_onlinereplay.csv"
LABELS_CSV = "NetworkIngestion/SSDP Flood_labels.csv"
OUTPUT_CSV = "Train and Test Files\\Kitsune Online Benchmarks\\SSDP_onlinereplay_labeled.csv"
LABEL_COL = 1  # 0 if labels are in first column, 1 if second column (after index)

print(f"[*] Adding labels from {LABELS_CSV} to {INPUT_CSV}")

with open(INPUT_CSV, 'r', newline='') as infile, \
     open(LABELS_CSV, 'r', newline='') as labelfile, \
     open(OUTPUT_CSV, 'w', newline='') as outfile:

    reader = csv.reader(infile)
    label_reader = csv.reader(labelfile)
    writer = csv.writer(outfile)

    # Skip headers
    header = next(reader)
    label_header = next(label_reader)  # Skip label file header too

    # Write header with label column first
    writer.writerow(['label'] + header)

    # Write data rows with labels from the original file
    for i, (row, label_row) in enumerate(zip(reader, label_reader)):
        label = label_row[LABEL_COL]
        writer.writerow([label] + row)

        if (i + 1) % 500000 == 0:
            print(f"[*] Processed {i + 1} rows...")

print(f"[*] Done! Output saved to {OUTPUT_CSV}")
