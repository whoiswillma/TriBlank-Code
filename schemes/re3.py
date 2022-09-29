import random

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from TriBlank import DEVICE
from docred_util import DOCRED_TRAIN, ALL_RELATION_IDS, format_example
from schemes.common import extract_labeled_edges


def all_index_triples(iter):
    for i in range(len(iter)):
        for j in range(i + 1, len(iter)):
            for k in range(j + 1, len(iter)):
                yield i, j, k


def is_related_set(example):
    related_indexes = set()
    label = example['labels']
    for head, tail in zip(label['head'], label['tail']):
        related_indexes.add((head, tail))
        related_indexes.add((tail, head))
    return related_indexes


def extract_triples(example):
    triples = set()
    is_related = is_related_set(example)
    for i, j, k in all_index_triples(example['vertexSet']):
        if (i, j) in is_related and (j, k) in is_related:
            triples.add((i, j, k))
    return triples


def extract_triway(example):
    triples = set()
    is_related = is_related_set(example)
    for i, j, k in all_index_triples(example['vertexSet']):
        if (i, j) in is_related and (j, k) in is_related and (i, k) in is_related:
            triples.add((i, j, k))
    return triples


def print_relation(example, index_pair):
    vertex_set = example['vertexSet']
    labels = example['labels']
    i, j = index_pair
    for head, relation, tail in zip(labels['head'], labels['relation_text'], labels['tail']):
        head_rep = vertex_set[head][0]['name']
        tail_rep = vertex_set[tail][0]['name']

        if (i, j) == (head, tail):
            print('{} {} {}'.format(head_rep, relation, tail_rep))

        elif (j, i) == (head, tail):
            print('{} {} {}'.format(tail_rep, relation, head_rep))


def print_triple_relation(example, index_triple):
    i, j, k = index_triple
    print_relation(example, (i, j))
    print_relation(example, (j, k))
    print_relation(example, (i, k))


def print_extract_triway(example):
    for index_triple in extract_triway(example):
        print(index_triple)
        print_triple_relation(example, index_triple)
        print()


def map_to_re3_dataset(dataset):
    re3_dataset = []

    for dataset_index, example in enumerate(tqdm(dataset)):
        labeled_edges = extract_labeled_edges(example)

        for i in range(len(example['vertexSet'])):
            for j in range(len(example['vertexSet'])):
                if i == j:
                    continue

                for k in range(len(example['vertexSet'])):
                    if i == k or j == k:
                        continue

                    if (i, j) not in labeled_edges or (j, k) not in labeled_edges:
                        continue

                    for rel_ij in labeled_edges[(i, j)]:
                        for rel_jk in labeled_edges[(j, k)]:
                            re3_dataset.append((
                                dataset_index,
                                i,
                                ALL_RELATION_IDS.index(rel_ij),
                                j,
                                ALL_RELATION_IDS.index(rel_jk),
                                k
                            ))

    return re3_dataset


def form_re3_batch(docred_dataset, re3_dataset, tokenizer, start_index, batch_size, blank_alpha=None):
    blank_alpha = blank_alpha or 0.7
    end_index = min(start_index + batch_size, len(re3_dataset))

    examples = [
        format_example(
            docred_dataset[i],
            [
                (e0, random.random() < blank_alpha),
                (e1, random.random() < blank_alpha),
                (e2, random.random() < blank_alpha)
            ]
        )
        for (i, e0, _, e1, _, e2) in re3_dataset[start_index:end_index]
    ]

    examples = tokenizer(examples, padding=True, return_tensors='pt', truncation=True).to(DEVICE)

    gold_01 = torch.tensor([x[2] for x in re3_dataset[start_index:end_index]]).to(DEVICE)
    gold_12 = torch.tensor([x[4] for x in re3_dataset[start_index:end_index]]).to(DEVICE)

    return examples, gold_01, gold_12


def iter_re3_batches(docred_dataset, re3_dataset, tokenizer, batch_size, blank_alpha=None):
    for start_index in tqdm(range(0, len(re3_dataset), batch_size)):
        yield form_re3_batch(
            docred_dataset,
            re3_dataset,
            tokenizer,
            start_index,
            batch_size,
            blank_alpha=blank_alpha
        )


def train_epoch_re3(
        model,
        tokenizer,
        optim,
        docred_dataset,
        re3_dataset,
        batch_size,
        max_grad_norm=None,
        blank_alpha=None
):
    model.train()

    for examples, gold_01, gold_12 in iter_re3_batches(
            docred_dataset,
            re3_dataset,
            tokenizer,
            batch_size,
            blank_alpha=blank_alpha
    ):
        optim.zero_grad()
        output = torch.transpose(model(examples), 0, 1)
        loss = F.nll_loss(output[0], gold_01) + F.nll_loss(output[2], gold_12)
        loss.backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()


if __name__ == '__main__':
    extract_triway(DOCRED_TRAIN[0])
