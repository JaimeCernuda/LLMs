import json
import re
from collections import defaultdict

# File paths
input_file = "./publications.ts"
output_file = "./publications.json"

# Read the TypeScript file
try:
    with open(input_file, "r", encoding="utf-8") as file:
        data_text = file.read()
except FileNotFoundError:
    print(f"❌ Error: File '{input_file}' not found!")
    exit()

# **Fix TypeScript-like format**
json_data = data_text.strip()

# **1. Ensure Object Keys are Quoted**
json_data = re.sub(r'(?<!")\b(\w+)\b(?=\s*:)', r'"\1"', json_data)  # Ensure all keys are quoted

# **2. Convert Single to Double Quotes in Keys Only, Avoiding Colons in Values**
json_data = re.sub(r"(?<=[:,{[])(?<!\\)'([^':]+)'(?!\\)(?=[},\]])", r'"\1"', json_data)  # Avoid modifying text values containing colons

# **3. Remove Trailing Commas in Lists and Objects**
json_data = re.sub(r",\s*([\]}])", r"\1", json_data)

# **4. Convert TypeScript `null` Values to JSON `null`**
json_data = json_data.replace("None", "null")

# **5. Ensure Proper Brackets**
if not json_data.startswith("["):
    json_data = "[" + json_data + "]"

# **Write the JSON to file (even if broken)**
try:
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(json_data)
    print(f"🚨 JSON written to '{output_file}' for debugging.")
except Exception as e:
    print(f"❌ Error writing to '{output_file}': {e}")
    exit()

# **Print first few lines for debugging**
try:
    with open(output_file, "r", encoding="utf-8") as file:
        preview = "".join(file.readlines()[:10])  # Show first 10 lines
        print("\n🔍 First few lines of 'publications.json':")
        print(preview)
except Exception as e:
    print(f"❌ Error reading '{output_file}': {e}")
    exit()

# **Now Try Parsing**
try:
    with open(output_file, "r", encoding="utf-8") as file:
        parsed_json = json.load(file)  # Ensure JSON is valid
    print(f"✅ Clean JSON successfully parsed from {output_file}")
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing error: {e}")
    print("Check the file manually for issues.")
    exit()

# **Extract Unique Authors and Their PDF Links**
authors_pdfs = defaultdict(set)

for pub in parsed_json:
    # Extract year from the date field
    date = pub.get("date", "")
    year_match = re.search(r"\b(\d{4})\b", date)
    year = int(year_match.group(1)) if year_match else 0

    if year >= 2015:
        pdf_link = pub.get("links", {}).get("pdf", None)
        if pdf_link:
            for author in pub.get("authors", []):
                authors_pdfs[author].add(pdf_link)

# **Print Results**
for author, pdfs in sorted(authors_pdfs.items()):
    print(f"Author: {author}")
    for pdf in pdfs:
        print(f"  PDF: {pdf}")
    print()
