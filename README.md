# TEMT: Text Encoder Meets Time

### Requirements

```sh
pip install -U sentence-transformers
```

### Data

```sh
https://drive.google.com/drive/folders/1poqQh_aioeMCJ99np7obFLO_s36H-AO5?usp=share_link
```

# Inductive split

```sh
python data/utils_from_blp.py drop_entities --file=../DATASETS/inductive/only-training/WIKIDATA12k/old-train.txt
python data/utils_from_blp.py drop_entities --file=../DATASETS/inductive/only-training/YAGO11k/old-train.txt
python data/utils_from_blp.py drop_entities --file=../DATASETS/inductive/all-triples/WIKIDATA12k/all-triples.txt
python data/utils_from_blp.py drop_entities --file=../DATASETS/inductive/all-triples/YAGO11k/all-triples.txt
python data/create_inductive_splits.py
```

### Experiments

# Time prediction

```sh
python time_prediction.py --data_dir "../DATASETS/YAGO11k"  --do_train --epochs 50 --batch 1024 --n_temporal_neg 128  --do_test --lr 0.001  --min_time -453 --max_time 2844 --margin 2 --save_model --save_to "yago11k_tp_model.pth" --use_descriptions
```

```sh
python time_prediction.py --data_dir "../DATASETS/WIKIDATA12k"  --do_train --epochs 50 --batch 1024 --n_temporal_neg 128 --do_test --lr 0.001  --margin 2 --save_model --save_to "wikidata12k_tp_model.pth" --use_descriptions
```

# Inductive

```sh
python time_prediction.py --data_dir "../DATASETS/inductive/all-triples/YAGO11k"  --do_train --epochs 50 --batch 1024 --n_temporal_neg 128  --do_test --lr 0.001 --min_time -453 --max_time 2844 --margin 2 --save_model --save_to  "ind_yago11k_tp_model.pth" --use_descriptions
```

```sh
python time_prediction.py --data_dir "../DATASETS/inductive/all-triples/WIKIDATA12k"  --do_train --epochs 50 --batch 1024 --n_temporal_neg 128 --do_test --lr 0.001  --margin 2 --save_model --save_to "ind_wikidata12k_tp_model.pth" --use_descriptions
```


### Acknowledgments
This code borrows from [KG-BERT](https://github.com/yao8839836/kg-bert), [BLP](https://github.com/dfdazac/blp), [TimePlex](https://github.com/dair-iitd/tkbi), and [Time2Box](https://github.com/ling-cai/Time2Box).
