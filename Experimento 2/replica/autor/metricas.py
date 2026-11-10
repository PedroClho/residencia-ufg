"""
Evaluation utilities for memory agent benchmarks.
Adopted from https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/utils/eval_utils.py
"""

# Cópia parcial: ver NOTICE.md para as funções copiadas e as mudanças feitas.

import string
import re
from collections import Counter
import nltk
import tiktoken



# ============================================================================
# TEXT NORMALIZATION AND SCORING UTILITIES
# ============================================================================

def normalize_answer(answer_text):
    """
    Normalize text for evaluation by removing articles, punctuation, and extra whitespace.
    
    Args:
        answer_text: The text to normalize
        
    Returns:
        Normalized text string
    """
    # Apply all normalization steps in sequence
    text = answer_text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    
    return text


def f1_score(prediction, ground_truth):
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # Handle special cases for yes/no/noanswer responses
    special_answers = {'yes', 'no', 'noanswer'}
    if ((normalized_prediction in special_answers or normalized_ground_truth in special_answers) and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    # Tokenize both texts and calculate token overlap
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common_tokens = sum(common_tokens.values())
    
    if num_common_tokens == 0:
        return ZERO_METRIC
    
    # Calculate precision, recall, and F1
    precision = num_common_tokens / len(prediction_tokens)
    recall = num_common_tokens / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall


def drqa_exact_match_score(prediction, ground_truth):
    """
    Check if prediction is an exact match with ground truth after normalization.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating exact match
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction, ground_truth):
    """
    Check if ground truth is a substring of the prediction after normalization.
    
    Args:
        prediction: The predicted text  
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating substring match
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def drqa_metric_max_over_ground_truths(metric_function, prediction, ground_truths):
    """
    Calculate the maximum score over multiple ground truth answers.
    
    Args:
        metric_function: Function to calculate score between prediction and single ground truth
        prediction: The predicted text
        ground_truths: List of ground truth answers (can be string, list, or nested list)
        
    Returns:
        Maximum score across all ground truths
    """
    # Normalize ground_truths to a flat list of strings
    if isinstance(ground_truths, str):
        ground_truth_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        # Flatten nested lists
        ground_truth_list = [gt for gt_sublist in ground_truths for gt in gt_sublist]
    else:
        ground_truth_list = ground_truths

    # Calculate score for each ground truth and return maximum
    return max(metric_function(prediction, gt) for gt in ground_truth_list)


def parse_output(output_text, answer_prefix="Answer:"):
    """
    Parse model output to extract the answer portion.
    
    Args:
        output_text: The complete model output
        answer_prefix: The prefix that indicates where the answer starts
        
    Returns:
        Extracted answer text or None if not found
    """
    # Try multiple patterns to extract the answer
    extraction_patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), 
        re.compile(r"(?:^)(.*)(?:\n|$)")
    ]
    
    for pattern in extraction_patterns:
        match = pattern.search(output_text)
        if match:
            extracted_text = match[1].strip()
            # Remove prefix again in case it was repeated
            clean_answer = re.sub(f'^{re.escape(answer_prefix)}', '', extracted_text, flags=re.IGNORECASE).strip()
            return clean_answer
    
    # Should rarely reach here, but return None if no pattern matches
    return None


# ============================================================================
# TEXT CHUNKING UTILITIES
# ============================================================================

def chunk_text_into_sentences(text, model_name="gpt-4o-mini", chunk_size=4096):
    """
    Split text into chunks of specified token size, preserving sentence boundaries.
    
    Args:
        text: The long text document to be split
        model_name: The tokenizer model name (default: gpt-4o-mini)
        chunk_size: Maximum number of tokens allowed per chunk
        
    Returns:
        List of text chunks, each within the specified token limit
    """
    # Ensure NLTK sentence tokenizer is available
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    # Initialize tokenizer with fallback
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Use fallback model if specified model is not recognized
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    text_chunks = []
    current_chunk_sentences = []
    current_chunk_token_count = 0

    for sentence in sentences:
        # Count tokens in current sentence
        sentence_tokens = encoding.encode(sentence, allowed_special={'<|endoftext|>'})
        sentence_token_count = len(sentence_tokens)
        
        # Check if adding this sentence would exceed chunk size
        if current_chunk_token_count + sentence_token_count > chunk_size:
            # Finalize current chunk and start new one
            text_chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_token_count += sentence_token_count
    
    # Add final chunk if it contains any sentences
    if current_chunk_sentences:
        text_chunks.append(" ".join(current_chunk_sentences))
    
    return text_chunks


def calculate_metrics(prediction, ground_truth_answers):
    """
    Calculate comprehensive metrics for prediction evaluation.
    
    Args:
        prediction: The predicted text
        ground_truth_answers: Ground truth answer(s) - can be string or list
        
    Returns:
        Dictionary of calculated metrics
    """
    # Calculate basic metrics using maximum over ground truths
    metrics = {
        "exact_match": drqa_metric_max_over_ground_truths(drqa_exact_match_score, prediction, ground_truth_answers),
        "f1": drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], prediction, ground_truth_answers),
        "substring_exact_match": drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, ground_truth_answers)
    }

    return metrics


def post_process(output, answer, dataset_config):
    """
    Apply dataset-specific post-processing to model outputs.
    
    Args:
        output: Model output dictionary
        answer: Ground truth answer
        dataset_config: Dataset configuration dictionary
        
    Returns:
        Tuple of (metrics_dict, additional_info_dict)
    """
    sub_dataset_name = dataset_config['sub_dataset']
    
    # Route to appropriate post-processing based on dataset type
    if 'icl' in sub_dataset_name:
        return _process_icl_dataset(output, answer)
    elif 'ruler' in sub_dataset_name or 'mermory_merging' in sub_dataset_name:
        return _process_ruler_memory_merging_dataset(output, answer, dataset_config)
    else:
        return default_post_process(output, answer)


def _process_icl_dataset(output, answer):
    """Process in-context learning dataset outputs."""
    prediction = output["output"]
    parsed_prediction = parse_output(prediction)
    metrics = calculate_metrics(parsed_prediction, answer)
    return metrics, {"parsed_output": parsed_prediction}



def _process_ruler_memory_merging_dataset(output, answer, dataset_config):
    """Process Ruler/Memory merging dataset outputs."""
    sub_dataset_name = dataset_config['sub_dataset']
    
    if "ruler_niah" in sub_dataset_name:
        prediction = output["output"]
        recall_score = sum([
            answer_element.lower() in prediction.lower() 
            for answer_element in answer
        ]) / len(answer)
        metrics = {"ruler_recall": recall_score}
        return metrics, {"parsed_output": prediction}
    else:
        return default_post_process(output, answer)


def default_post_process(output, answer):
    """
    Default post-processing function for model outputs.
    
    Args:
        output: Model output dictionary
        answer: Ground truth answer
        
    Returns:
        Tuple of (metrics_dict, additional_info_dict)
    """
    prediction = output["output"]
    metrics = calculate_metrics(prediction, answer)
    
    # Try parsing output and take maximum scores
    parsed_prediction = parse_output(prediction)
    if parsed_prediction is not None:
        parsed_metrics = calculate_metrics(parsed_prediction, answer)
        metrics = {metric_name: max(original_score, parsed_metrics[metric_name]) 
                  for metric_name, original_score in metrics.items()}
    
    return metrics, {"parsed_output": parsed_prediction}
