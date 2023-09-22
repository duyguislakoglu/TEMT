import numpy as np
import torch
from eval.interval_prediction_methods import greedy_coalescing
from eval.interval_metrics import gaeiou_score, aeiou_score, giou_score
import pandas as pd
from eval.helper import get_thresholds
import logging
import pickle

def eval_tp(model, device, test_loader, time_vectors, time_range, ground_truth_intervals, k=10, args=None, constant_threshold=0.65):
    model.eval()
    aeiou_scores_atk = 0
    gaeiou_scores_atk = 0
    giou_scores_atk = 0

    aeiou_scores_at1 = 0
    gaeiou_scores_at1 = 0
    giou_scores_at1 = 0

    print(len(ground_truth_intervals))
    print(len(test_loader))

    with torch.no_grad():
        for j, batch in enumerate(test_loader):

            batch = tuple(t.to(device) for t in batch)

            (time_encoding, label, triple_encoding) = batch

            scores = []
            triple_encoding = triple_encoding.reshape((triple_encoding.shape[0], triple_encoding.shape[2]))

            for time_vector in time_vectors:

                # TODO: It will change when the batch size changes
                time_vector = time_vector.reshape((1, -1))
                time_vector = torch.from_numpy(time_vector).float()
                time_vector = time_vector.to(device)

                score = model(triple_encoding, torch.Tensor(time_vector))
                # TODO: Will change when the batch size change
                scores.append(score[0][0].cpu())

            scores = np.array(scores)
            print(scores)

            #TODO: this will take batch size instead of 1.
            scores =  torch.from_numpy(scores.reshape((1, scores.shape[0])))

            probs = torch.nn.functional.softmax(scores.to(dtype=torch.float64), dim=-1)
            print(probs)

            # TODO: Get this from threshold method in helper
            thresholds = [constant_threshold] * len(probs)
            #with open("../thresholds.dat", "rb") as f:
            #    thresholds = pickle.load(f)

            preds_min, preds_max = greedy_coalescing(probs, thresholds, k=k)

            gold_min = int(ground_truth_intervals[j][0])
            gold_max = int(ground_truth_intervals[j][1])

            # TODO: Scoring methods do not work with one input so putting one more.
            gold_min = np.array([gold_min, gold_min])
            gold_min = torch.Tensor(gold_min.reshape((-1,1)))
            gold_max = np.array([gold_max, gold_max])
            gold_max = torch.Tensor(gold_max.reshape((-1,1)))

            best_aeiou_atk = 0
            best_gaeiou_atk = 0
            best_giou_atk = 0

            best_aeiou_at1 = 0
            best_gaeiou_at1 = 0
            best_giou_at1 = 0

            for i in range(k):
                # TODO: this will not be 0 when batch changes
                pred_min = preds_min[0][i]
                pred_max = preds_max[0][i]

                pred_min = np.array([time_range[int(pred_min)],time_range[int(pred_min)]])
                pred_min = torch.Tensor(pred_min.reshape((-1,1)))
                pred_max = np.array([time_range[int(pred_max)],time_range[int(pred_max)]])
                pred_max = torch.Tensor(pred_max.reshape((-1,1)))

                print(pred_min[0], pred_max[0])
                print(gold_min[0], gold_max[0])

                # TODO: this will not be 0 when batch changes
                aeiou = aeiou_score(pred_min, pred_max, gold_min, gold_max)[0]
                gaeiou = gaeiou_score(pred_min, pred_max, gold_min, gold_max)[0]
                giou = giou_score(pred_min, pred_max, gold_min, gold_max)[0]

                if i == 0:
                    aeiou_scores_at1 += aeiou
                    gaeiou_scores_at1 += gaeiou
                    giou_scores_at1 += giou

                if aeiou > best_aeiou_atk:
                    best_aeiou_atk = aeiou

                if gaeiou > best_gaeiou_atk:
                    best_gaeiou_atk = gaeiou

                if giou > best_giou_atk:
                    best_giou_atk = giou

            aeiou_scores_atk += best_aeiou_atk
            gaeiou_scores_atk += best_gaeiou_atk
            giou_scores_atk += best_giou_atk

            cnt = j+1

            print("Current giou score @1: " + str(giou_scores_at1/cnt))
            print("Current aeiou score @1: " + str(aeiou_scores_at1/cnt))
            print("Current gaeiou score @1: "   + str(gaeiou_scores_at1/cnt))

            print(("Current giou_scores @{}: ").format(k) + str(giou_scores_atk/cnt))
            print(("Current aeiou_scores @{}: ").format(k) + str(aeiou_scores_atk/cnt))
            print(("Current gaeiou_scores @{}: ").format(k) + str(gaeiou_scores_atk/cnt))


        print("Final giou score @1: " + str(giou_scores_at1/cnt))
        print("Final aeiou score @1: " + str(aeiou_scores_at1/cnt))
        print("Final gaeiou score @1: "   + str(gaeiou_scores_at1/cnt))

        print(("Final giou score: @{}: ").format(k) + str(giou_scores_atk/cnt))
        print(("Final aeiou score: @{}: ").format(k) + str(aeiou_scores_atk/cnt))
        print(("Final gaeiou score: @{}: ").format(k) + str(gaeiou_scores_atk/cnt))

        file_prefix = str(args.data_dir) + "_" + "nwords" + str(args.number_of_words) + "_"
        file_prefix += "tn" +str(args.n_temporal_neg) + "_" + "ctn" +str(args.n_corrupted_triple) + "_"
        file_prefix += "td" + str(args.time_dimension) + "_" + "ep" +str(args.epochs) + "_" + "lr" +str(args.lr) + "_" + "mr" + str(args.margin) + "thr" + str(args.constant_threshold)

        f = open(file_prefix + 'TP.txt','a')
        f.write("Final giou score @1: " + str(giou_scores_at1/cnt)+ '\n')
        f.write("Final aeiou score @1: " + str(aeiou_scores_at1/cnt)+ '\n')
        f.write("Final gaeiou score @1: " + str(gaeiou_scores_at1/cnt)+ '\n')

        f.write(("Final giou score: @{}: ").format(k) + str(giou_scores_atk/cnt) + '\n')
        f.write(("Final aeiou score: @{}: ").format(k) + str(aeiou_scores_atk/cnt) + '\n')
        f.write(("Final gaeiou score: @{}: ").format(k) + str(gaeiou_scores_atk/cnt)+ '\n')
        f.close()
