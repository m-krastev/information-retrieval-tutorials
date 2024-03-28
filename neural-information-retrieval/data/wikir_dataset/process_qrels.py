import json

# Read the JSONL file
with open("dev_qrels.jsonl", "r") as f:
    lines = f.readlines()

# Initialize an empty dictionary for the results
results = {}

# Process each line
for line in lines:
    # Parse the JSON object
    obj = json.loads(line)

    # Get the query ID and doc ID
    query_id = obj["query_id"]
    doc_id = obj["doc_id"]

    # If the query ID is not in the results yet, add it
    if query_id not in results:
        results[query_id] = {}

    # Add the doc ID to the query's dictionary
    results[query_id][doc_id] = 1

# Write the results to a JSON file
with open("dev_qrels.json", "w") as f:
    json.dump(results, f)