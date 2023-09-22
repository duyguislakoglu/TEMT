import os
with open(os.path.join('', "../DATASETS/inductive/all-triples/WIKIDATA12k/ind-train.tsv"), 'r') as f:
    tr = set()
    lines = f.readlines()
    for line in lines:
        s,p,o = line.split("\t")
        tr.update([" ".join([s,p,o[:-1]])])
with open(os.path.join('', "../DATASETS/inductive/all-triples/WIKIDATA12k/ind-dev.tsv"), 'r') as f:
    dev = set()
    lines = f.readlines()
    for line in lines:
        s,p,o = line.split("\t")
        dev.update([" ".join([s,p,o[:-1]])])
with open(os.path.join('', "../DATASETS/inductive/all-triples/WIKIDATA12k/ind-test.tsv"), 'r') as f:
    test = set()
    lines = f.readlines()
    for line in lines:
        s,p,o = line.split("\t")
        test.update([" ".join([s,p,o[:-1]])])

with open('../DATASETS/inductive/all-triples/WIKIDATA12k/train.txt','w') as file1:
    with open('../DATASETS/inductive/all-triples/WIKIDATA12k/valid.txt','w') as file2:
        with open('../DATASETS/inductive/all-triples/WIKIDATA12k/test.txt','w') as file3:
            with open(os.path.join('', "../DATASETS/inductive/all-triples/WIKIDATA12k/all-triples.txt"), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    s,p,o = line.split("\t")[:3]
                    if " ".join([s,p,o]) in tr:
                        file1.write(line)
                    elif " ".join([s,p,o]) in dev:
                        file2.write(line)
                    else:
                        file3.write(line)

with open(os.path.join('', "../DATASETS/inductive/all-triples/YAGO11k/ind-train.tsv"), 'r') as f:
    tr = set()
    lines = f.readlines()
    for line in lines:
        s,p,o = line.split("\t")
        tr.update([" ".join([s,p,o[:-1]])])
with open(os.path.join('', "../DATASETS/inductive/all-triples/YAGO11k/ind-dev.tsv"), 'r') as f:
    dev = set()
    lines = f.readlines()
    for line in lines:
        s,p,o = line.split("\t")
        dev.update([" ".join([s,p,o[:-1]])])
with open(os.path.join('', "../DATASETS/inductive/all-triples/YAGO11k/ind-test.tsv"), 'r') as f:
    test = set()
    lines = f.readlines()
    for line in lines:
        s,p,o = line.split("\t")
        test.update([" ".join([s,p,o[:-1]])])

with open('../DATASETS/inductive/all-triples/YAGO11k/train.txt','w') as file1:
    with open('../DATASETS/inductive/all-triples/YAGO11k/valid.txt','w') as file2:
        with open('../DATASETS/inductive/all-triples/YAGO11k/test.txt','w') as file3:

            with open(os.path.join('', "../DATASETS/inductive/all-triples/YAGO11k/all-triples.txt"), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    s,p,o = line.split("\t")[:3]
                    if " ".join([s,p,o]) in tr:
                        file1.write(line)
                    elif " ".join([s,p,o]) in dev:
                        file2.write(line)
                    else:
                        file3.write(line)
