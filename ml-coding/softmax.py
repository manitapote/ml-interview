import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
  max_arg = np.max(logits, axis=-1, keepdims=True)
  logits = logits - max_arg
  exp = np.exp(logits)

  sum_total = np.sum(exp, axis=-1, keepdims=True)
  sum_total = np.clip(sum_total, 1e-6, None)

  return exp / sum_total

def categorical_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the average Categorical Cross-Entropy Loss given raw logits and 1-hot targets.

    Parameters:
    -----------
    logits : np.ndarray
        Shape (N, C), unnormalized logit scores for N samples across C classes.
    targets : np.ndarray
        Shape (N, C), one-hot encoded ground truth labels.

    Returns:
    --------
    float
        Average loss scalar across all N samples.
    """

    # sum_c p_i log(q_i)
    softmax_probs = softmax(logits)
    log_probs = -np.log(softmax_probs)

    log_probs = np.sum(targets * log_probs, axis=-1)

    return round(log_probs.mean(), 4)

if __name__ == "__main__":
    # Test Case 1: Standard Batch (N=2 samples, C=3 classes)
    logits = np.array([
        [2.0, 1.0, 0.1],  # Softmax probabilities: ~[0.6590, 0.2424, 0.0986]
        [0.5, 3.5, 1.0]   # Softmax probabilities: ~[0.0440, 0.8821, 0.0739]
    ])
    
    targets = np.array([
        [1, 0, 0],  # True label: Class 0 -> Target prob = ~0.6590 -> loss1 = -log(0.6590) ≈ 0.4170
        [0, 1, 0]   # True label: Class 1 -> Target prob = ~0.8821 -> loss2 = -log(0.8821) ≈ 0.1254
    ])

    # Calculated average loss: (0.4170 + 0.1254) / 2 = ~0.2712
    expected_loss = 0.2712
    
    output_loss = categorical_cross_entropy(logits, targets)
    
    print(f"Sample Logits Shape:  {logits.shape}")
    print(f"Sample Targets Shape: {targets.shape}")
    print(f"Computed Loss:       {output_loss}")
    print(f"Expected Loss:       {expected_loss}")
    
    if output_loss is not None:
        assert np.isclose(output_loss, expected_loss, atol=1e-4), "❌ Test Case 1 Failed"
        print("✅ Test Case 1 Passed!")