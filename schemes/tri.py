import random

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from TriBlank import DEVICE
from docred_util import ALL_RELATION_IDS, format_example
from schemes.common import extract_labeled_edges


def map_to_tri_dataset(dataset):
    tri_dataset = []

    for dataset_index, example in enumerate(tqdm(dataset)):
        labeled_edges = extract_labeled_edges(example)

        for i in range(len(example['vertexSet'])):
            for j in range(len(example['vertexSet'])):
                if i == j:
                    continue

                for k in range(len(example['vertexSet'])):
                    if i == k or j == k:
                        continue

                    if (i, j) not in labeled_edges or (i, k) not in labeled_edges or (j, k) not in labeled_edges:
                        continue

                    for rel_ij in labeled_edges[(i, j)]:
                        for rel_ik in labeled_edges[(i, k)]:
                            for rel_jk in labeled_edges[(j, k)]:
                                tri_dataset.append((
                                    dataset_index,
                                    i,
                                    j,
                                    k,
                                    ALL_RELATION_IDS.index(rel_ij),
                                    ALL_RELATION_IDS.index(rel_ik),
                                    ALL_RELATION_IDS.index(rel_jk),
                                ))

    return tri_dataset


def form_tri_batch(docred_dataset, tri_dataset, tokenizer, start_index, batch_size, blank_alpha=None):
    blank_alpha = blank_alpha or 0.7
    end_index = min(start_index + batch_size, len(tri_dataset))

    examples = [
        format_example(
            docred_dataset[i],
            [
                (e0, random.random() < blank_alpha),
                (e1, random.random() < blank_alpha),
                (e2, random.random() < blank_alpha)
            ]
        )
        for (i, e0, e1, e2, _, _, _) in tri_dataset[start_index:end_index]
    ]

    examples = tokenizer(examples, padding=True, return_tensors='pt', truncation=True).to(DEVICE)

    gold_01 = torch.tensor([x[4] for x in tri_dataset[start_index:end_index]]).to(DEVICE)
    gold_02 = torch.tensor([x[5] for x in tri_dataset[start_index:end_index]]).to(DEVICE)
    gold_12 = torch.tensor([x[6] for x in tri_dataset[start_index:end_index]]).to(DEVICE)

    return examples, gold_01, gold_02, gold_12


def iter_tri_batches(docred_dataset, tri_dataset, tokenizer, batch_size, blank_alpha=None):
    for start_index in tqdm(range(0, len(tri_dataset), batch_size)):
        yield form_tri_batch(
            docred_dataset,
            tri_dataset,
            tokenizer,
            start_index,
            batch_size,
            blank_alpha=blank_alpha
        )


def train_epoch_tri(
        model,
        tokenizer,
        optim,
        docred_dataset,
        tri_dataset,
        batch_size,
        max_grad_norm=None,
        blank_alpha=None
):
    model.train()

    for examples, gold_01, gold_02, gold_12 in iter_tri_batches(
            docred_dataset,
            tri_dataset,
            tokenizer,
            batch_size,
            blank_alpha=blank_alpha
    ):

        optim.zero_grad()
        output = torch.transpose(model(examples), 0, 1)
        loss = F.nll_loss(output[0], gold_01) + F.nll_loss(output[1], gold_02) + F.nll_loss(output[2], gold_12)
        loss.backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()


def eval_contingency_table_tri(model, tokenizer, dataset, tri_dataset, blank=False):
    model.eval()

    contingency_table = [[0] * len(ALL_RELATION_IDS) for _ in range(len(ALL_RELATION_IDS))]

    with torch.no_grad():
        for i, e0, e1, e2, rel_01, rel_02, rel_12 in tqdm(tri_dataset):
            input = tokenizer(
                [format_example(dataset[i], [(e0, blank), (e1, blank), (e2, blank)])],
                padding=True,
                return_tensors='pt',
                truncation=True
            ).to(DEVICE)

            output = model(input).squeeze(0)

            pred = torch.argmax(output[0]).item()
            contingency_table[rel_01][pred] += 1

            pred = torch.argmax(output[1]).item()
            contingency_table[rel_02][pred] += 1

            pred = torch.argmax(output[2]).item()
            contingency_table[rel_12][pred] += 1

    return contingency_table
