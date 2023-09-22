import torch
import numpy
from models import CrossEncoderWithTime
from collections import defaultdict

# From Time2Box
def get_thresholds2(valid_time_scores_info, test_time_scores_info, aggr='mean', verbose=False):
    rel_prob_list = defaultdict(list)

    rel_interval_len = defaultdict(list)

    for idx, info in enumerate(valid_time_scores_info): # enumerate all the possibilities
        s, r, o = info[0][:3] ## change the format for the output
        start, end = info[0][3], info[0][4]
        # t = fact[3:] #

        probs = torch.nn.functional.softmax(info[-1], dim=-1)
        # print("Probabilities shape:", probs.shape)

        prob_sum = 0.0
        for i in range(start, end + 1):
            prob_sum += probs[0,i]

        rel_prob_list[r].append(float(prob_sum))
        rel_interval_len[r].append(end - start + 1)

    # print("Num relations:", len(rel_prob_list))

    rel_thresh = {}

    for key, val in rel_prob_list.items():
        # rel_thresh[key]=numpy.median(numpy.array(val))
        if aggr == 'mean':
            rel_thresh[key] = numpy.mean(numpy.array(val))
        elif aggr == 'median':
            rel_thresh[key] = numpy.median(numpy.array(val))
        else:
            raise Exception('Unknown aggregate {} for thresholds'.format(aggr))
    print(rel_thresh)
    '''
    if(verbose):
        print("\nRelations thresholds:")
        for key,val in rel_thresh.items():
            print(key,val)
            print(id2rel[ktrain.reverse_relation_map[key]])
            print(numpy.mean(numpy.array(rel_interval_len[key])), numpy.std(numpy.array(rel_interval_len[key])), numpy.median(numpy.array(rel_interval_len[key])))
            print("Freq:",len(rel_interval_len[key]))

            print('\n')
    '''

    thresholds = torch.zeros(len(test_time_scores_info))

    thresh_list = [i for _, i in rel_thresh.items()]
    mean_thresh = sum(thresh_list) / len(thresh_list) # this is used for cases when there are relations in test dataset but not in valid dataset.
    # print("Mean threshold:{}\n".format(mean_thresh))

    for idx, fact in enumerate(test_time_scores_info):
        r = info[0][1]
        if r in rel_thresh:
            thresholds[idx] = rel_thresh[r]
        else:
            # print("{} relation not in dict".format(r))
            thresholds[idx] = mean_thresh

    return thresholds ## this is the threshold for each fact



# Test relations are easy to get. Just parse.
# Valid relations should be doubled after parsing because they are all closed.
# Valid loader should create data loader: no neg, only_ends
def get_thresholds(load_from, valid_loader, valid_relations, test_relations):
    # e.g. valid_relations = [1,1,4,4,5,5,4,4]
    # e.g. val_preds = [0.5, 0.5, 0.8, 0.8, 0.6, 0.6, 0.8, 0.8]
    # e.g. test_relations = [1,2,3,4,5,1]

    # 0. Load a model
    # 1. Get scores for all validation instances
    # 2. Take means for each relation as thresholds
    # 3. Return a list (threshold for each fact in test set) with length len(test_set)

    # Since all facts in the valid test is closed interval, preds[i] and preds[i+1] will correspond to same fact.
    # valid_relations handle it

    # 0. Load a model
    model = CrossEncoderWithTime()
    model.load_state_dict(torch.load(load_from))
    model.to(device)

    # 1. Get scores for all validation instances
    model.eval()

    val_preds = []

    with torch.no_grad():
        for j, batch in enumerate(valid_loader):
            batch = tuple(t.to(device) for t in batch)
            (time_encoding, label, triple_encoding) = batch
            triple_encoding = triple_encoding.reshape((triple_encoding.shape[0], triple_encoding.shape[2]))

            logits = model(triple_encoding, time_encoding)

            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)

            else:
                batch_logits = logits.detach().cpu().numpy()
                val_preds[0] = np.append(preds[0], batch_logits, axis=0)

    rel_predictions = {}
    rel_predictions = defaultdict(lambda:[],rel_predictions)


    for (j, relation) in enumerate(valid_relations):
        rel_predictions[relation] = rel_predictions[relation].append(val_preds[j])

    # 2. Take means for each relation as thresholds
    for k in rel_predictions:
        rel_predictions[k] = numpy.mean(numpy.array(rel_predictions[k]))

    thresholds = torch.zeros(len(test_relations))
    for (i,rel) in enumarate(test_relations):
        if rel in rel_predictions:
            threshods[rel] = rel_predictions[rel]
        else:
            threshods[rel] = 0.65

    # 3. Return a list with length len(test_set)
    return thresholds
