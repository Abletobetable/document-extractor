"""
functions for preparing features
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/content/ru-document-tokenizer')

PAD_ON_RIGHT = tokenizer.padding_side == "right"

# some hyperparameters
MAX_LENGTH = 1000 # length of document in batch
STRIDE = 128 # overlap between features
N_BEST_SIZE = 20 # number of best answers to keep for choosing final answer
MAX_ANSWER_LENGTH = 100

def prepare_train_features(examples):
    """
    preprocess dataset: 
    tokenize documents and cut long examples if needed
    """
    
    # remove left whitespace
    examples["label"] = [q.lstrip() for q in examples["label"]]

    # Tokenize our examples with truncation and padding,
    # but keep the overflows using a stride.
    # So when document is to long it truncated in several examples with overlaps
    tokenized_examples = tokenizer(
        examples["label" if PAD_ON_RIGHT else "text"],
        examples["text" if PAD_ON_RIGHT else "label"],
        truncation="only_second" if PAD_ON_RIGHT else "only_first",
        max_length=MAX_LENGTH,
        stride=STRIDE, 
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # since one example might be splitted in several examples,
    # I need to map from generated example and origanal example
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # the offset mappings will give us a map:
    # from token to character position in the original context
    # this will help us compute the start_positions and end_positions
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # label those examples
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # label impossible answers with the index of the CLS token
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # grab the sequence corresponding to that example
        # (to know what is the context and what is the question)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # one example can give several spans,
        # this is the index of the example containing this span of text
        sample_index = sample_mapping[i]
        answers = examples["extracted_part"][sample_index]

        # if no answers are given, set the cls_index as answer
        if answers["text"][0] == '':
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # start/end character index of the answer in the text
            start_char = answers["answer_start"][0]
            end_char = answers["answer_end"][0]

            # start token index of the current span in the text
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if PAD_ON_RIGHT else 0):
                token_start_index += 1

            # end token index of the current span in the text
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if PAD_ON_RIGHT else 0):
                token_end_index -= 1

            # detect if the answer is out of the span
            # (in which case this feature is labeled with the CLS index)
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):

                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # else move the token_start_index and token_end_index to the two ends of the answer
                # we could go after the last offset if the answer is the last word
                while (token_start_index < len(offsets)
                      and offsets[token_start_index][0] <= start_char):

                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples):
    """
    preprocess dataset: 
    tokenize documents and cut long examples if needed
    """

    # remove left whitespace
    examples["label"] = [q.lstrip() for q in examples["label"]]

    # tokenize our examples with truncation and padding,
    # but keep the overflows using a stride
    # so when document is to long it truncated in several examples with overlaps
    tokenized_examples = tokenizer(
        examples["label" if PAD_ON_RIGHT else "text"],
        examples["text" if PAD_ON_RIGHT else "label"],
        truncation="only_second" if PAD_ON_RIGHT else "only_first",
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # since one example might be splitted in several examples,
    # I need to map from generated example and origanal example
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # keep the example_id that gave this feature and store the offset mappings
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # grab the sequence corresponding to that example
        # (to know what is the context and what is the question)
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if PAD_ON_RIGHT else 0

        # one example can give several spans,
        # this is the index of the example containing this span of text
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # set to None the offset_mapping that are not part of the text
        # so it's easy to determine if a token position is part of the text or not
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples