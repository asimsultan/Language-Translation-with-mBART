import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import MBartTokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_data(examples, tokenizer, max_length):
    inputs = tokenizer(examples['translation']['en'], truncation=True, padding='max_length', max_length=max_length)
    targets = tokenizer(examples['translation']['fr'], truncation=True, padding='max_length', max_length=max_length)

    inputs['labels'] = targets['input_ids']
    return inputs


def create_data_loader(dataset, batch_size, sampler):
    data_sampler = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
    return data_loader
