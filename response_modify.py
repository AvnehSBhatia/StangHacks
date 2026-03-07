from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pretrained model once (do this at startup)
model = SentenceTransformer("all-MiniLM-L6-v2")

def sentence_to_vector(sentence: str) -> np.ndarray:
    """
    Convert a sentence into a dense vector embedding.
    """
    embedding = model.encode(sentence)
    return embedding  # shape: (384,)

# Example usage
# if __name__ == "__main__":
#     s = "I like integrals."
#     vec = sentence_to_vector(s)
#     print("Vector shape:", vec.shape)
#     print("First 5 values:", vec[:5])

def to_matrix(strings: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Convert a list of strings into a matrix of vectors.
    Each row corresponds to one string. Uses batch encoding for efficiency.
    """
    if not strings:
        return np.array([]).reshape(0, 384)
    embeddings = model.encode(strings, batch_size=batch_size, show_progress_bar=False)
    return np.array(embeddings)


def vectorize_pair(
    questions_1: list[str], answers_1: list[str],
    questions_2: list[str], answers_2: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorize two user pairs for training. Returns (Q1, A1, Q2, A2) as (10, 384) arrays.
    Use to_matrix internally; call with 10 questions and 10 answers per user.
    """
    all_strings = questions_1 + answers_1 + questions_2 + answers_2
    all_embeddings = model.encode(all_strings, batch_size=64, show_progress_bar=False)
    n = len(questions_1)
    Q1 = np.array(all_embeddings[0:n])
    A1 = np.array(all_embeddings[n : 2 * n])
    Q2 = np.array(all_embeddings[2 * n : 3 * n])
    A2 = np.array(all_embeddings[3 * n : 4 * n])
    return Q1, A1, Q2, A2

if __name__ == "__main__":
    responses = [
        "Response 1",
        "Response 2",
        "Response 3",
        "Response 4",
        "Response 5",
        "Response 6",
        "Response 7",
        "Response 8",
        "Response 9",
        "Response 10",
    ]
    
    questions = [
        "Question 1",
        "Question 2",
        "Question 3",
        "Question 4",
        "Question 5",
        "Question 6",
        "Question 7",
        "Question 8",
        "Question 9",
        "Question 10",
    ]
    responses_matrix = to_matrix(responses)
    questions_matrix = to_matrix(questions)
    print("Responses: Number of vectors (rows):", len(responses_matrix))        # 10
    print("Responses: Vector length (columns):", len(responses_matrix[0]))      # 256 by default
    print("Questions: Number of vectors (rows):", len(questions_matrix))        # 10
    print("Questions: Vector length (columns):", len(questions_matrix[0]))      # 256 by default