import csv

# Input and output file names
input_file = "unique_authors.txt"
output_file = "unique_authors.csv"

# Open the input file and process it
authors_data = {}
with open(input_file, "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if line.strip() or line == ""]  # Preserve empty lines

    for line in lines:
        if line.startswith("Author:"):
            author = line.replace("Author: ", "").strip()
            continue
        elif line.startswith("Link: "):
            link = line.replace("Link: ", "").strip()
            authors_data[author] = link
        else:
            raise Exception("Unknown line type")

# Write to a CSV file
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Author", "Link"])  # Write header
    for key, value in authors_data.items():
       csv_writer.writerow([key, value])

print(f"CSV file '{output_file}' has been created successfully.")
