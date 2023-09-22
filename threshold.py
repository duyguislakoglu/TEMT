import torch
from models import CrossEncoderWithTime
from data_processor import TKGProcessor
import pickle
from dataset import TensorDatasetWithMoreNegatives
from torch.utils.data import DataLoader
import numpy as np
import os
import collections

model = CrossEncoderWithTime(time_dimension=128)
model.load_state_dict(torch.load("../BEST/model.pth"))
#model.to(device)

min_time=19
max_time=2020

processor = TKGProcessor("../DATASETS/WIKIDATA12k", "tp", "test", min_time=min_time, max_time=max_time, time_dimension=128)

#val_examples = processor.get_dev_examples("../DATASETS/WIKIDATA12k")
#val_features = processor.convert_examples_to_features(val_examples)

#with open("../val_features.dat", "wb") as f:
#      pickle.dump(val_features, f)
with open("../val_features.dat", "rb") as f:
    val_features = pickle.load(f)

all_time_encoding = torch.tensor([f.time_encoding for f in val_features])
all_label = torch.tensor([f.label for f in val_features])
all_triple_encoding = torch.tensor([f.triple_encoding for f in val_features])

val_data = TensorDatasetWithMoreNegatives(all_time_encoding,
                all_label, all_triple_encoding, number_of_negatives=0)

valid_loader = DataLoader(val_data, batch_size=1, shuffle=False)
time_vectors = processor.time_encoder.time_vectors

model = CrossEncoderWithTime(time_dimension=128)
model.load_state_dict(torch.load("model1.pth"))

model.eval()

def get_relations(data_dir, mode):

    relations = []

    path = os.path.join(data_dir, "intervals/{}.txt".format(mode))

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("\t")
            relations.append(line[1])

    return relations

val_preds = []
ground_truth_intervals = processor.get_ground_truth_val_intervals("../DATASETS/WIKIDATA12k")
valid_relations = get_relations("../DATASETS/WIKIDATA12k", "valid")

s = 0

rel_predictions = {}
rel_predictions = collections.defaultdict(lambda:[],rel_predictions)

with torch.no_grad():
    for j, batch in enumerate(valid_loader):
        #batch = tuple(t.to(device) for t in batch)
        (time_encoding, label, triple_encoding) = batch
        triple_encoding = triple_encoding.reshape((triple_encoding.shape[0], triple_encoding.shape[2]))

        scores = []

        for time_vector in time_vectors:

            # TODO: It will change when the batch size changes
            time_vector = time_vector.reshape((1, -1))
            time_vector = torch.from_numpy(time_vector).float()
            #time_vector = time_vector.to(device)

            score = model(triple_encoding, torch.Tensor(time_vector))
            # TODO: Will change when the batch size change
            scores.append(score[0][0].cpu())

        scores = np.array(scores)
        #print(scores)

        #TODO: this will take batch size instead of 1.
        scores =  torch.from_numpy(scores.reshape((1, scores.shape[0])))

        probs = torch.nn.functional.softmax(scores.to(dtype=torch.float64), dim=-1)

        #print(probs)
        #print(probs[:,-10])
        #print(probs[:,-11])
        (g_start, g_end) = ground_truth_intervals[j]
        #print(sum(probs[0][int(g_start)-min_time:int(g_end)-min_time+1]))
        if valid_relations[j] in rel_predictions.keys():
            rel_predictions[valid_relations[j]].append(sum(probs[0][int(g_start)-min_time:int(g_end)-min_time+1]))
        else:
            rel_predictions[valid_relations[j]] = [sum(probs[0][int(g_start)-min_time:int(g_end)-min_time+1])]
# Take means for each relation as thresholds
for k in rel_predictions:
    rel_predictions[k] = np.mean(np.array(rel_predictions[k]))

print(rel_predictions)

test_relations = get_relations("../DATASETS/WIKIDATA12k", "test")

thresholds = torch.zeros(len(test_relations))
for (i,rel) in enumerate(test_relations):
    if rel in rel_predictions:
        thresholds[i] = rel_predictions[rel]
    else:
        thresholds[i] = 0.65

print(thresholds)
print(thresholds.shape)

with open("thresholds.dat", "wb") as f:
      pickle.dump(thresholds, f)
