from requirement_1.r1_methods_v2 import run_similarity_analysis

# File paths
source_path = r"D:\appathon_personal\field_mapper\requirement_1\vendor_input_format.csv"
target_path = r"D:\appathon_personal\field_mapper\requirement_1\customer_standard_format.csv"
output_file = r"mappings_v2.csv"
model_name = "all-MiniLM-L6-v2"

# Model choice - Change this to any SentenceTransformer model
model_name = "paraphrase-MiniLM-L6-v2" 
# mpnet = "all-mpnet-base-v2"

# Run the analysis
run_similarity_analysis(source_path, target_path, model_name, output_file, filter_highest_only=False)