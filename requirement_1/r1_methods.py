import pandas as pd
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def read_and_clean_data(input_path):
    """
    Reads a CSV file, removes unused fields, and returns a list of business names.
    """
    data = pd.read_csv(input_path, encoding="ISO-8859-1")
    data_filtered = data[~data.astype(str).apply(lambda row: row.str.contains("Field is not currently used", 
                                                                              case=False, na=False)).any(axis=1)]
    return data_filtered['Business Name'].dropna().tolist()  # Drop NaN values

def compute_cosine_similarity(vendor_list, customer_list, model):
    """
    Computes cosine similarity between customer and vendor field names using the specified model.
    """
    vendor_embeddings = model.encode(vendor_list, convert_to_tensor=True)
    customer_embeddings = model.encode(customer_list, convert_to_tensor=True)
    return util.pytorch_cos_sim(customer_embeddings, vendor_embeddings)  # Similarity matrix

def find_best_matches(customer_list, vendor_list, similarity_matrix):
    """
    Finds the best match for each customer field from the vendor list using Cosine Similarity.
    """
    best_matches = []
    for i, customer_field in enumerate(customer_list):
        best_match_idx = similarity_matrix[i].argmax().item()
        best_match_target = vendor_list[best_match_idx]
        best_match_score = similarity_matrix[i][best_match_idx].item()

        best_matches.append((customer_field, best_match_target, round(best_match_score, 4)))
    return best_matches

def save_results_to_csv(results, output_filename):
    """
    Writes the best match results to a CSV file with Cosine Similarity scores.
    """
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Customer Field", "Vendor Field", "Cosine Similarity"])
        writer.writerows(results)
    print(f"Results saved to {output_filename}")

def run_similarity_analysis(source_path, target_path, model_name, output_filename):
    """
    Runs the similarity analysis using a specified model.
    
    Parameters:
    - source_path (str): Path to the vendor input CSV file.
    - target_path (str): Path to the customer input CSV file.
    - model_name (str): SentenceTransformer model to use.
    - output_filename (str): File where results will be saved.
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    vendor_fields = read_and_clean_data(source_path)
    customer_fields = read_and_clean_data(target_path)

    similarity_matrix = compute_cosine_similarity(vendor_fields, customer_fields, model)

    best_matches = find_best_matches(customer_fields, vendor_fields, similarity_matrix)

    save_results_to_csv(best_matches, output_filename)

    return best_matches  # Returns the results if needed elsewhere



