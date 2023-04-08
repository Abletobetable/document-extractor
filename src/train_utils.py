"""
functions for train, test and log
"""

import os
import pprint
import numpy as np
import pandas as pd
from typing import Literal
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import get_scheduler
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('/content/ru-document-tokenizer')
PAD_ON_RIGHT = tokenizer.padding_side == "right"

def trainer(model, train_loader, valid_loader,
            optimizer, scheduler, cfg):
    """
    Parameters
    ----------
        trainer:
            iterating through epochs and call train_epoch
        cfg (dict()):
            keys():
                count_of_epoch:
                    numer of epochs
                batch_size:
                    batch size
                lr:
                    learing rate for optimizer
                model_name:
                    name of model, needed for saving weights in right folder
                device:
                    use cpu or gpu for training
            
        train_loader:
            train split
        valid_loader:
            validation split
        model:
            model for training
        
        optimizer:
            optimizer
        scheduler:
            learning rate scheduler for optimizer
    """

    wandb.watch(model, loss_function, log="all", log_freq=10)

    min_valid_loss = np.inf

    # in this folder will save model weights
    if not os.path.exists(f'/content/model_weights/{cfg["model_name"]}'):
        os.makedirs(f'/content/model_weights/{cfg["model_name"]}')

    # main loop
    for e in tqdm(range(cfg['count_of_epoch']), desc='epochs'):

        # train
        epoch_loss = train_epoch(train_generator=train_loader,
                                 model=model,
                                 optimizer=optimizer,
                                 device=os.environ['device'])

        # validation
        valid_loss = 0.0
        model.eval()
        valid_loss, valid_f1 = tester(model=model,
                                        test_loader=valid_loader,
                                        device=os.environ['device'])

        scheduler.step()

        # log things
        trainer_log(epoch_loss, valid_loss, valid_f1, e,
                    optimizer.param_groups[0]['lr'], cfg)

        # saving models
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(model.state_dict(),
                       f'/content/model_weights/{cfg["model_name"]}/saved_model_{e}.pth')
            wandb.log_artifact(f'/content/model_weights/{cfg["model_name"]}/saved_model_{e}.pth',
                                name=f'saved_model_{e}', type='model')
        print()

def train_epoch(train_generator, model,
                optimizer, device):
    """
    iterating through batches inside batch generator 
    and call train_on_batch
    Parameters
    ----------
        train_generator:
            batch generator for training
        model:
            model for training
        optimizer:
            optimizer
    Return
    ------
        mean loss over epoch
    """
    epoch_loss = 0
    total = 0
    for batch in train_generator:
        batch_loss = train_on_batch(model,
                                    batch['pixel_values'], batch['label'],
                                    optimizer, loss_function, device)

        epoch_loss += batch_loss*len(batch['pixel_values'])
        total += len(batch['pixel_values'])

    return epoch_loss/total

def train_on_batch(model, batch,
                   optimizer, device):
    """
    train on single batch
    Parameters
    ----------
        model:
            model for traning
        batch:
            batch features and targets
        optimizer:
            optimizer
    Return
    ------
        loss on the batch
    """
    model.train()
    optimizer.zero_grad()

    batch = batch.to(device)
    output = model(**batch)

    # extraction loss is the sum of a Cross-Entropy for the start/end positions
    loss = output.loss
    loss.backward()

    optimizer.step()
    return loss.cpu().item()

def tester(model, test_loader, loss_function = None, 
           print_stats=False, device='cpu'):
    """
    testing or validating on provided dataset
    also if needed print some statistics
    Parameters
    ----------
        model:
            model for traning
        test_loader:
            dataset for testing or validating
        loss_function:
            criterion for calculating loss function on validation
        print_stats (bool):
            rather print statistics or not
        device (str):
            use cpu or gpu for testing
    Return
    ------
        mean loss over validation dataset and f1_score or just f1_score
    """
    pred = []
    real = []
    loss = 0
    model.eval()
    for batch in test_loader:

        x_batch = batch['pixel_values'].to(device)
        with torch.no_grad():
            output = model(x_batch)

            if loss_function is not None:
                loss += loss_function(output, batch['label'].to(device))

        pred.extend(torch.argmax(output, dim=-1).cpu().numpy().tolist())
        real.extend(batch['label'].cpu().numpy().tolist())

    F1 = f1_score(real, pred, average='weighted')

    if print_stats:
        print(F1)

    if loss_function is not None:
        return loss.cpu().item()/len(test_loader), F1
    else:
        wandb.log({'test_f1': F1})
        return F1

def trainer_log(train_loss, valid_loss, valid_f1, epoch, lr, cfg):
    """
    make logging
    """
    wandb.log({'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_f1': valid_f1, 
                'epoch': epoch, 
                'learning_rate': lr})

    print(f'train loss on {str(epoch).zfill(3)} epoch: {train_loss:.6f} with lr: {lr:.10f}')
    print(f'valid loss on {str(epoch).zfill(3)} epoch: {valid_loss:.6f}')
    print(f'valid f1 score: {valid_f1:.2f}')

def prepare_train_features(examples):
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

def train_pipeline(dataset, cfg, saved_model=None,
                   to_train=True, to_test=True):
    """
    run training and/or testing process
    Parameters
    ----------
        dataset:
            splitted dataset that has part for train and validation
        saved_model:
            path to saved checkpoint to resume training
            or to test saved model
        cfg:
            parameters config
        to_train (bool):
            if True train, else only test
        to_test (bool):
            if True test, else only train
    Return
    ------
        trained or tested model
    """

    def build_model(model, saved_model=None):
        """
        initialise model
        """

        p = cfg['dropout']

        if cfg['model_name'] == 'mobilebert':
            model_checkpoint = 'aware-ai/mobilebert-squadv2'

        model = AutoModelForQuestionAnswering.from_pretrained(
            model_checkpoint,
            max_position_embeddings = 1000 if cfg['max_length'] == 1000 else 512,
            ignore_mismatched_sizes=True)

        model = model.to(device)

        if saved_model is not None:
            model.load_state_dict(torch.load(saved_model, 
                                  map_location=torch.device(device)))
            model = model.to(device)

        return model

    def prepare_features(dataset, cfg):

        MAX_LENGTH = cfg['max_length']
        STRIDE = cfg['stride']

        tokenized_dataset = dataset.map(
            prepare_train_features, 
            batched=True, 
            remove_columns=dataset['train'].column_names,
            load_from_cache_file=False)

        return tokenized_dataset

    def make(model, tokenized_dataset):
        """
        make dataloaders, init optimizers and criterions
        """

        tokenized_dataset = tokenized_dataset.with_format("torch")

        trainloader = DataLoader(tokenized_dataset['train'], 
                                 batch_size=cfg['batch_size'],
                                 shuffle=True, num_workers=2)
        trainloader = DataLoader(tokenized_dataset['test'], 
                                 batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=2)

        optimizer = AdamW(model.parameters(), lr=cfg['lr'])

        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, 
            num_warmup_steps=0)

        return trainloader, validloader, criterion, optimizer, scheduler

    # pretty print dict()
    pretty_print = pprint.PrettyPrinter()

    print('config:')
    pretty_print.pprint(cfg)
    print()
    print('running on device:', device, '\n')

    # build the model
    model = build_model(model, saved_model)

    # prepare datasets
    tokenized_dataset = prepare_features(dataset, cfg)

    # dataloaders and optimization
    trainloader, validloader,  \
        criterion, optimizer, scheduler = make(model, tokenized_dataset)

    if to_train:
        trainer(model, trainloader, validloader,
                criterion, optimizer, scheduler, cfg)

    if to_test:
        tester(model, validloader, print_stats=True, 
                device=device)

    return model
