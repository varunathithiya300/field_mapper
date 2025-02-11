# Standard Library Imports
from pathlib import Path
import logging

# Third-Party Library Imports
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from mteb import MTEB

# Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(Model training started)s")

def trainSentenceTransformer(csv_path: str, model_name: str, output_path: str, batch_size: int = 8, epochs: int = 3):
    """
    Trains a SentenceTransformer model using data from a CSV file.

    Args:
        csv_path (str): Path to the training data CSV file.
        model_name (str): Name of the pre-trained SentenceTransformer model.
        output_path (str): Path to save the trained model.
        batch_size (int, optional): Batch size for training. Default is 8.
        epochs (int, optional): Number of training epochs. Default is 3.
    """

    logging.info("Loading data from CSV...")
    data_for_training = Path(csv_path)
    df = pd.read_csv(data_for_training, encoding="ISO-8859-1")

    logging.info(f"Loaded {len(df)} records from {csv_path}")

    training_data = [
        InputExample(texts=[row["Field Name"], row["Definition"]], label=float(row["Similarity"]))
        for _, row in df.iterrows()
    ]

    logging.info("Initializing the model...")
    model = SentenceTransformer(model_name)

    training_data_loader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    logging.info("Starting training...")
    model.fit(
        train_objectives=[(training_data_loader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_path
    )

    model.save(output_path)
    logging.info(f"Model saved successfully to '{output_path}'")

# Example usage
if __name__ == "__main__":
    trainSentenceTransformer(
        csv_path=r"D:\appathon_personal\field_mapper\requirement_1_developments\vendor_input_training.csv",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        output_path="ehr_trained_model"
    )



# model = SentenceTransformer("domain_sentence_transformer")

# sentences = ["biological sample", "a specimen collected for testing"]
# embeddings = model.encode(sentences)

# from sklearn.metrics.pairwise import cosine_similarity
# similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
# print(f"Similarity Score: {similarity[0][0]:.4f}")

evaluation = MTEB(tasks=["STSBenchmark"])
results = evaluation.run("ehr_trained_model")
print(results)