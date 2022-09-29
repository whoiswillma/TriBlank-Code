def extract_labeled_edges(example):
    labels = example['labels']
    labeled_edges = {}

    for head, relation, tail in zip(labels['head'], labels['relation_id'], labels['tail']):
        key = (head, tail)

        if key not in labeled_edges:
            labeled_edges[key] = []

        labeled_edges[key].append(relation)

    return labeled_edges
