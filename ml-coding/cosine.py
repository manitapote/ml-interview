
import numpy as np

def cosine_similarity_matrix(query_embeddings: np.ndarray, 
                             document_embeddings: np.ndarray
                             ) -> np.ndarray:
    """
    Computes the pairwise cosine similarity matrix between queries and documents.

    Parameters:
    -----------
    query_embeddings : np.ndarray
        Shape (N, D), where N is the number of query vectors.
    document_embeddings : np.ndarray
        Shape (M, D), where M is the number of document vectors.

    Returns:
    --------
    np.ndarray
        Shape (N, M), containing pairwise cosine similarities.
    """
    #normalize quires and documents sqrt
    q_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    d_norm = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    q_norm = np.where(q_norm == 0, 1 , q_norm)
    d_norm = np.where(d_norm == 0, 1, d_norm)

    q_e_n = query_embeddings/q_norm
    d_e_n = document_embeddings/d_norm

    #a.b = (n, d), (m, d)
    cosine = np.dot(q_e_n, d_e_n.T)

    return cosine


if __name__ == "__main__":
    # Test Case 1: Simple 2x3 vectors
    queries = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ])
    
    docs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 0.0]  # Zero-vector edge case
    ])

    result = cosine_similarity_matrix(queries, docs)
    print("Output Matrix:\n", np.round(result, 4))
    
    # Expected shape: (2, 3)
    # Expected values approx:
    # [[1.    , 0.    , 0.    ],
    #  [0.    , 1.    , 0.    ]]