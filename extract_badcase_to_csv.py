import re
import csv

input_file = "error_log.txt"
output_file = "badcase_extract.csv"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Extract all cases using regex
pattern = re.compile(
    r"Now complete the following example:(.*?)Gold Domain: (.*?)\nPredicted Domain: (.*?)\n", re.DOTALL
)
matches = pattern.findall(content)

with open(output_file, "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["context", "gold_domain", "predicted_domain"])
    for dialog, gold_domain, pred_domain in matches:
        # Remove extra whitespace
        dialog = dialog.strip().replace('\n', ' ')
        gold_domain = gold_domain.strip()
        pred_domain = pred_domain.strip()
        writer.writerow([dialog, gold_domain, pred_domain])

print(f"Extracted {len(matches)} cases, saved to {output_file}") 