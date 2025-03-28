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
        # Find the choice that best matches the prediction
        best_match = -1
        best_score = -1
        
        for i, choice in enumerate(choices_list):
            # Simple string matching - for better results, use semantic similarity
            if choice.lower() in pred.lower():
                score = len(choice) / len(pred)  # Longer matches are better
                if score > best_score:
                    best_score = score
                    best_match = i
        
        if best_match == answer_idx:
            correct += 1
    
    return correct / len(predictions)
