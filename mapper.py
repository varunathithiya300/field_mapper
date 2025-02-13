from mapping_template.r1_methods import run_similarity_analysis
from pathlib import Path

# File paths
source_path = Path(r"D:\appathon_personal\field_mapper\mapping_template\vendor_input_format.csv")
target_path = Path(r"D:\appathon_personal\field_mapper\mapping_template\customer_standard_format.csv")

output_file = r"mappings.csv"
model_name = "all-MiniLM-L6-v2"

# Run the analysis
run_similarity_analysis(source_path, target_path, model_name, output_file)
