import numpy as np
from rouge import Rouge

def exact_match_accuracy(predictions, references):
    """Calculate exact match accuracy."""
    correct = 0
    for pred, ref in zip(predictions, references):
        if pred.lower() == ref.lower():
            correct += 1
    return correct / len(predictions)

def contains_answer(predictions, references):
    """Check if reference is contained in prediction."""
    correct = 0
    for pred, ref in zip(predictions, references):
        if ref.lower() in pred.lower():
            correct += 1
    return correct / len(predictions)

def calculate_rouge(predictions, references):
    """Calculate ROUGE scores."""
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores

def choice_accuracy(predictions, choices, answer_indices):
    """Calculate accuracy for multiple-choice questions."""
    correct = 0
    for pred, choices_list, answer_idx in zip(predictions, choices, answer_indices):
        # Clean up the prediction - get just the letter
        pred = pred.strip().upper()
        if len(pred) > 0:
            # If the prediction starts with a letter A-D, use that
            if pred[0] in "ABCD":
                selected_idx = ord(pred[0]) - ord('A')
                if selected_idx == answer_idx:
                    correct += 1
    
    return correct / len(predictions)
