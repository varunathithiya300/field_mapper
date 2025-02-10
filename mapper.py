from requirement_1.r1_methods import run_similarity_analysis

# File paths
source_path = r"D:\appathon_parser\AKA300\requirement_1\vendor_input_format.csv"
target_path = r"D:\appathon_parser\AKA300\requirement_1\customer_standard_format.csv"
output_file = r"D:\appathon_parser\AKA300\requirement_1\mappings.csv"
model_name = "all-MiniLM-L6-v2"

# Model choice
model_name = "paraphrase-MiniLM-L6-v2"  # Change this to any SentenceTransformer model

# Run the analysis
run_similarity_analysis(source_path, target_path, model_name, output_file)
