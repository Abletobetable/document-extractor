"""
functions for choosing the best prediction
"""

import collections
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/content/ru-document-tokenizer')

def postprocess_predictions(
    examples,
    features,
    raw_predictions, 
    n_best_size = 20,
    max_answer_length = 100):
    """
    Check answers for correctness
        Greedy predictions my not be valid: some of them might be impossible,
        so I check if extracted text is possible
    
    Also convert start/end tokens in start/end chars and extract predicted text
    """

    all_start_logits, all_end_logits = raw_predictions
    # build a map example to its corresponding features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # result dictionaries
    predictions = collections.OrderedDict()

    # loop over all the examples
    for example_index, example in enumerate(tqdm(examples)):
        # indices of the features associated to the current example
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["text"]
        # loop through all the features associated to the current example
        for feature_index in feature_indices:
            # grab the predictions of the model for this feature
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # map some the positions in our logits to
            # span of texts in the original context
            offset_mapping = features[feature_index]["offset_mapping"]

            # update minimum null prediction
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score > feature_null_score:
                min_null_score = feature_null_score

            # pick only n_best_size logits
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # check for impossible answers
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # check for impossible answers
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # if there is non-null prediction,
            # create a empty prediction to avoid failure
            best_answer = {"text": "", "score": 0.0}
        
        # pick final answer: the best one or the null answer
        answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        predictions[example["id"]] = answer

    return predictions
