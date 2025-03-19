import json
from collections import defaultdict

# File path
input_file = "./publications_base.json"

# Read the JSON file
try:
    with open(input_file, "r", encoding="utf-8") as file:
        publications = json.load(file)
except FileNotFoundError:
    print(f"❌ Error: File '{input_file}' not found!")
    exit()
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing error: {e}")
    exit()

# Extract unique authors with one PDF or poster link
authors_links = {}

for pub in publications:
    authors = pub.get("authors", [])
    links = pub.get("links", {})
    
    # Prioritize PDF, fall back to poster
    link = links.get("pdf") or links.get("poster")
    
    if link:
        for author in authors:
            if author not in authors_links:  # Store only the first found link
                authors_links[author] = link

# Print results
for author, link in sorted(authors_links.items()):
    print(f"Author: {author}")
    print(f"  Link: {link}")
    print()
