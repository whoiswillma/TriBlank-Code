import gc
import random

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from TriBlank import DEVICE
from docred_util import format_example, ALL_RELATION_IDS


def map_to_re2_dataset(dataset):
    re2_dataset = []

    for i, example in enumerate(tqdm(dataset)):
        labels = example['labels']

        for head, rel, tail in zip(labels['head'], labels['relation_id'], labels['tail']):
            re2_dataset.append((i, head, ALL_RELATION_IDS.index(rel), tail))

    return re2_dataset


def form_re2_batch(docred_dataset, re2_dataset, tokenizer, start_index, batch_size, blank_alpha=None):
    blank_alpha = blank_alpha or 0.7
    end_index = min(start_index + batch_size, len(re2_dataset))

    examples = [
        format_example(
            docred_dataset[i],
            [
                (e0, random.random() < blank_alpha),
                (e1, random.random() < blank_alpha)
            ]
        )
        for (i, e0, _, e1) in re2_dataset[start_index:end_index]
    ]

    examples = tokenizer(examples, padding=True, return_tensors='pt', truncation=True).to(DEVICE)

    gold = torch.tensor([x[2] for x in re2_dataset[start_index:end_index]]).to(DEVICE)

    return (examples, gold)


def iter_re2_batches(docred_dataset, re2_dataset, tokenizer, batch_size, blank_alpha=None):
    for start_index in tqdm(range(0, len(re2_dataset), batch_size)):
        yield form_re2_batch(
            docred_dataset,
            re2_dataset,
            tokenizer,
            start_index,
            batch_size,
            blank_alpha=blank_alpha
        )


def train_epoch_re2(
        model,
        tokenizer,
        optim,
        docred_dataset,
        re2_dataset,
        batch_size,
        max_grad_norm=None,
        blank_alpha=None
):
    model.train()

    for examples, gold in iter_re2_batches(
            docred_dataset,
            re2_dataset,
            tokenizer,
            batch_size,
            blank_alpha=blank_alpha
    ):
        optim.zero_grad()
        output = torch.transpose(model(examples), 0, 1)
        loss = F.nll_loss(output[0], gold)
        loss.backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()

        del output
        del loss
        torch.cuda.empty_cache()
        gc.collect()


def eval_contingency_table_re2(model, tokenizer, dataset, re2_dataset, blank=False):
    model.eval()

    contingency_table = [[0] * len(ALL_RELATION_IDS) for _ in range(len(ALL_RELATION_IDS))]

    with torch.no_grad():
        for i, e0, rel, e1 in tqdm(re2_dataset):
            input = tokenizer(
                [format_example(dataset[i], [(e0, blank), (e1, blank)])],
                padding=True,
                return_tensors='pt',
                truncation=True
            ).to(DEVICE)

            output = model(input).squeeze(0)
            pred = torch.argmax(output[0]).item()
            contingency_table[rel][pred] += 1

    return contingency_table
