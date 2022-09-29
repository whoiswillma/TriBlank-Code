import datasets
from tqdm import tqdm

from TriBlank import ENT_END, ENT_BGN, ENT_BLN

DOCRED = datasets.load_dataset('docred')
DOCRED_TRAIN = DOCRED['train_annotated']
DOCRED_VALID = DOCRED['validation']
DOCRED_TEST = DOCRED['test']


def get_all_relation_ids(dataset):
    ids = set()
    for example in tqdm(dataset, leave=False):
        ids |= set(example['labels']['relation_id'])
    ids = list(ids)
    ids.sort()
    return ids


def relation_ids_to_text(dataset):
    rel2txt = {}
    for example in tqdm(dataset, leave=False):
        labels = example['labels']
        for rel, txt in zip(labels['relation_id'], labels['relation_text']):
            rel2txt[rel] = txt
    return rel2txt


ALL_RELATION_IDS = get_all_relation_ids(DOCRED_TRAIN)
RELATION_ID_TO_TEXT = relation_ids_to_text(DOCRED_TRAIN)

assert set(get_all_relation_ids(DOCRED['validation'])).issubset(set(ALL_RELATION_IDS))
assert set(get_all_relation_ids(DOCRED['test'])).issubset(set(ALL_RELATION_IDS))
assert set(get_all_relation_ids(DOCRED['train_distant'])).issubset(set(ALL_RELATION_IDS))


def format_example(example, eid_and_bln):
    # flatten example['sents']
    sents = []
    sentence_offsets = [0]
    for sentence in example['sents']:
        sents += sentence
        sentence_offsets.append(sentence_offsets[-1] + len(sentence))

    entities = example['vertexSet']

    fmt = lambda entity, index: (
        sentence_offsets[entity['sent_id']] + entity['pos'][0],
        sentence_offsets[entity['sent_id']] + entity['pos'][1],
        index
    )

    # an associated list from (start, end) positions to entity index { 0, 1, 2 }
    pos_to_ek = []
    for i, (eid, _) in enumerate(eid_and_bln):
        for entity in entities[eid]:
            pos_to_ek.append(fmt(entity, i))
    pos_to_ek.sort(key=lambda x: x[0])

    # remove overlap
    indexes = set()
    for i, (bgn, end, _) in reversed(list(enumerate(pos_to_ek))):
        if len(set(range(bgn, end)) & indexes) != 0:
            del pos_to_ek[i]
        else:
            indexes |= set(range(bgn, end))

    # assert no overlap
    # indexes = set()
    # for bgn, end, _ in pos_to_ek:
    #     assert len(set(range(bgn, end)) & indexes) == 0, (bgn, end, indexes)
    #     indexes |= set(range(bgn, end))

    # insert entity markers
    for bgn, end, ei in reversed(pos_to_ek):
        sents.insert(end, ENT_END[ei])
        sents.insert(bgn, ENT_BGN[ei])

        if eid_and_bln[ei][1]:
            # blank the entity
            sents[bgn + 1] = ENT_BLN
            for i in reversed(range(bgn + 2, end + 1)):
                del sents[i]

    return ' '.join(sents)


if __name__ == '__main__':
    print(format_example(DOCRED_TRAIN[25], [(0, False), (1, False), (2, False)]))
