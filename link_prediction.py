import pandas as pd
import numpy as np
import torch
import os
from models import CrossEncoderWithTime
from dataset import TensorDatasetWithMoreNegatives
import torch.nn as nn
from torch.nn import MarginRankingLoss
import argparse
import pickle
from data_processor import TKGProcessor
from torch.utils.data import DataLoader, TensorDataset
import logging

os.environ['CUDA_VISIBLE_DEVICES']= '0'

def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                         help='input batch size for training (default: 64)')
     parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                         help='input batch size for testing (default: 1000)')
     parser.add_argument('--epochs', type=int, default=50, metavar='N',
                         help='number of epochs to train (default: 14)')
     parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                         help='learning rate (default: 0.1)')
     parser.add_argument('--margin', type=float, default=2,
                         help='margin (default:2)')

     parser.add_argument('--save_model',
                         action='store_true',
                         help='For Saving the current model')
     parser.add_argument("--save_to",
                        default="model.pth",
                        type=str,
                        help="The directory for saving the model. save/to/path/model.pth")
     parser.add_argument('--pretrained', action='store_true',
                         help='For laoding the current Model')
     parser.add_argument('--load_from', default='', type=str, help="Pretrained model checkpoint")


     parser.add_argument('--no_cuda', action='store_true', default=False,
                         help='disables CUDA training')
     parser.add_argument('--seed', type=int, default=1, metavar='S',
                         help='random seed (default: 1)')


     parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
     parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test.")
     parser.add_argument("--saved_features",
                        action='store_true',
                        help="Whether to use saved features.")

     parser.add_argument("--use_descriptions",
                        action='store_true',
                        help="Using the descriptions.")

     parser.add_argument('--min_time', type=int, default=19, metavar='N',
                         help='Minimum in time range')
     parser.add_argument('--max_time', type=int, default=2020, metavar='N',
                         help='Max in time range')
     parser.add_argument('--time_dimension', type=int, default=64, metavar='N',
                           help='Dimension of time vectors.')
     parser.add_argument('--n_temporal_neg', type=int, default=0, metavar='N',
                                   help='Number of temporal negatives.')
     parser.add_argument('--n_corrupted_triple', type=int, default=0, metavar='N',
                                   help='Number of non-temporal negatives.')
     parser.add_argument('--number_of_words', type=int, default=15, metavar='N',
                                   help='Number of words in the descriptions.')


     args = parser.parse_args()

     use_cuda = not args.no_cuda and torch.cuda.is_available()
     #torch.manual_seed(args.seed)
     device = torch.device("cuda" if use_cuda else "cpu")

     if args.do_train:
         ################ M O D E L ######################

         processor = TKGProcessor(args.data_dir, "lp", "train", min_time=args.min_time, max_time=args.max_time, time_dimension=args.time_dimension)
         processor.n_temporal_neg = args.n_temporal_neg
         processor.n_corrupted_triple = args.n_corrupted_triple

         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

         model = CrossEncoderWithTime(time_dimension=args.time_dimension)

         model.to(device)
         optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

         #loss_fn = nn.BCEsLoss()
         loss_fn = nn.MarginRankingLoss(margin=args.margin)
         model.train()

         ################ D A T A ##################
         data_prefix = str(args.data_dir.split("/")[-1]) + "_" + str(args.number_of_words) + "_"
         data_prefix += str(args.n_temporal_neg) + "_" + str(args.n_corrupted_triple) + "_"
         data_prefix += str(args.time_dimension)

         if args.saved_features:

             with open("../"+str(data_prefix)+".dat", "rb") as f:
                 train_features = pickle.load(f)
         else:
             train_examples = processor.get_train_examples(args.data_dir)
             train_features = processor.convert_examples_to_features(train_examples, use_descriptions=args.use_descriptions)

             with open("../"+str(data_prefix)+".dat", "wb") as f:
                  pickle.dump(train_features, f)

         all_time_encoding = torch.tensor([f.time_encoding for f in train_features])
         all_label = torch.tensor([f.label for f in train_features])
         all_triple_encoding = torch.tensor([f.triple_encoding for f in train_features])

         train_data = TensorDatasetWithMoreNegatives(all_time_encoding,
                        all_label, all_triple_encoding, number_of_negatives=processor.n_temporal_neg + processor.n_corrupted_triple)

         train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

         #########################################

         losses = []
         correct = 0
         for i in range(args.epochs):
             for j, batch in enumerate(train_loader):
                 batch = tuple(t.to(device) for t in batch)
                 (time_encoding, label, triple_encoding) = batch

                 (pos_triple, negs_triple) = triple_encoding[:,0,:], triple_encoding[:,1:,:]
                 (pos_time, negs_time) = time_encoding[:,0,:], time_encoding[:,1:,:]
                 (pos_label, negs_label) = label[:,0],  label[:,1:]

                 pos_time = torch.Tensor(pos_time.float())

                 for k in range(negs_time.shape[1]):
                     neg_time = torch.Tensor(negs_time[:,k,:].float())

                     output1 = model(pos_triple, pos_time)
                     output2 = model(negs_triple[:,k,:], neg_time)

                     loss = loss_fn(output1, output2, torch.from_numpy(np.ones(pos_label.shape)).reshape(-1,1).to(device))

                     optimizer.zero_grad()
                     loss.backward()
                     optimizer.step()

             losses.append(loss.item())
             print("epoch {}\tloss : {}".format(i,loss))

             if args.save_model:
                 torch.save(model.state_dict(), args.save_to)

     if args.do_test:
         data_prefix = str(args.data_dir.split("/")[-1]) + "_" + str(args.number_of_words) + "_"
         data_prefix += str(args.n_temporal_neg) + "_" + str(args.n_corrupted_triple) + "_"
         data_prefix += str(args.time_dimension)


         processor = TKGProcessor(args.data_dir, task="lp", mode="test", min_time=args.min_time, max_time=args.max_time, time_dimension=args.time_dimension)

         model.eval()
         model.to(device)

         # run link prediction

         ranks = []
         ranks_left = []
         ranks_right = []

         hits_left = []
         hits_right = []
         hits = []

         top_ten_hit_count = 0

         for i in range(10):
             hits_left.append([])
             hits_right.append([])
             hits.append([])

         test_quadruples_lines = processor._read_tsv(os.path.join(args.data_dir, "test.txt"))

        # Time-aware filtering
         for (i,test_quadruple_line) in enumerate(test_quadruples_lines):

             #######################################################################################################################################
             time_interval = processor.interval_from_text(test_quadruple_line[3], test_quadruple_line[4])

             rank_for_each_time_point = []

             for time_point in time_interval:
                 print("The current time point is:" + str(time_point))
                 # We already saved years in time_interval so it is safe to overwrite test_quadruple_line
                 # Make it single year
                 test_quadruple_line[3] = str(time_point)
                 test_quadruple_line[4] = str(time_point)

                 head_corrupt_lines = [test_quadruple_line]

                 for tmp_head in processor.entities:
                     if tmp_head != test_quadruple_line[0]:
                         tmp_triple = (tmp_head, test_quadruple_line[1], test_quadruple_line[2])

                         if not processor.in_tkg_lp(tmp_triple, test_quadruple_line[3]):
                             head_corrupt_lines.append((tmp_head, test_quadruple_line[1], test_quadruple_line[2], test_quadruple_line[3], test_quadruple_line[4]))
                         else:
                             print("The candidate is filtered out.")

                 print("The length of head_corrupt_lines is:" + str(len(head_corrupt_lines)))

                 tmp_examples = processor.create_examples(head_corrupt_lines, args.data_dir, "only_ends")
                 tmp_features = processor.convert_examples_to_features(tmp_examples, use_descriptions=args.use_descriptions)

                 all_time_encoding = torch.tensor([f.time_encoding for f in tmp_features])
                 all_label = torch.tensor([f.label for f in tmp_features])
                 all_triple_encoding = torch.tensor([f.triple_encoding for f in tmp_features])

                 test_data = TensorDatasetWithMoreNegatives(all_time_encoding,
                            all_label, all_triple_encoding, number_of_negatives=processor.n_temporal_neg + processor.n_corrupted_triple)

                 test_loader = DataLoader(test_data, batch_size=len(head_corrupt_lines), shuffle=False)

                 preds = []

                 with torch.no_grad():
                     for j, batch in enumerate(test_loader,0):
                         batch = tuple(t.to(device) for t in batch)
                         (time_encoding, label, triple_encoding) = batch
                         #prob = model(subject_text_encoding, predicate_text_encoding, object_text_encoding, time_encoding)
                         #preds.append(prob[0][0])
                         triple_encoding = triple_encoding.reshape((triple_encoding.shape[0], triple_encoding.shape[2]))
                         triple_encoding = triple_encoding.to(torch.float32)

                         time_encoding = time_encoding.reshape((time_encoding.shape[0], time_encoding.shape[2]))
                         time_encoding = time_encoding.to(torch.float32)

                         logits = model(triple_encoding, time_encoding)

                         if len(preds) == 0:
                             batch_logits = logits.detach().cpu().numpy()
                             preds.append(batch_logits)

                         else:
                             batch_logits = logits.detach().cpu().numpy()
                             preds[0] = np.append(preds[0], batch_logits, axis=0)

                 rel_values = np.array(preds)
                 rel_values = torch.tensor(rel_values)

                 print("Predictions and the shape:")
                 print(rel_values, rel_values.shape)

                 #####
                 _, argsort1 = torch.sort(rel_values[0,:,0], descending=True)
                 rank_time_point = np.where(argsort1 == 0)[0][0]

                 print("The current time point is:" + str(time_point))
                 print("The rank for this time point is:" + str(rank_time_point))

                 rank_for_each_time_point.append(rank_time_point)

             print("rank_for_each_time_point:" + str(rank_for_each_time_point))
             rank1 = np.mean(np.array(rank_for_each_time_point))
             print("mean:" + str(rank1))
             ranks.append(rank1+1)
             ranks_left.append(rank1+1)
             if rank1 < 10:
                 top_ten_hit_count += 1

             #######################################################################################################################################

             rank_for_each_time_point = []

             for time_point in time_interval:
                 print("The current time point is:" + str(time_point))

                 # We already saved years in time_interval so it is safe to overwrite test_quadruple_line
                 # Make it single year
                 test_quadruple_line[3] = str(time_point)
                 test_quadruple_line[4] = str(time_point)

                 tail_corrupt_lines = [test_quadruple_line]

                 for tmp_tail in processor.entities:
                     if tmp_tail != test_quadruple_line[2]:
                         tmp_triple = (test_quadruple_line[0], test_quadruple_line[1], tmp_tail)
                         if not processor.in_tkg_lp(tmp_triple, test_quadruple_line[3]):
                             tail_corrupt_lines.append((test_quadruple_line[0], test_quadruple_line[1], tmp_tail, test_quadruple_line[3], test_quadruple_line[4]))
                         else:
                             print("The candidate is filtered out.")
                 print("The length of tail_corrupt_lines is:" + str(len(tail_corrupt_lines)))

                 tmp_examples = processor.create_examples(tail_corrupt_lines, args.data_dir, "only_ends")
                 tmp_features = processor.convert_examples_to_features(tmp_examples, use_descriptions=args.use_descriptions)

                 all_time_encoding = torch.tensor([f.time_encoding for f in tmp_features])
                 all_label = torch.tensor([f.label for f in tmp_features])
                 all_triple_encoding = torch.tensor([f.triple_encoding for f in tmp_features])

                 test_data = TensorDatasetWithMoreNegatives(all_time_encoding,
                            all_label, all_triple_encoding, number_of_negatives=processor.n_temporal_neg + processor.n_corrupted_triple)

                 test_loader = DataLoader(test_data, batch_size=len(tail_corrupt_lines), shuffle=False)

                 preds = []

                 with torch.no_grad():
                     for j, batch in enumerate(test_loader,0):
                         batch = tuple(t.to(device) for t in batch)
                         (time_encoding, label, triple_encoding) = batch
                         #prob = model(subject_text_encoding, predicate_text_encoding, object_text_encoding, time_encoding)
                         #preds.append(prob[0][0])
                         triple_encoding = triple_encoding.reshape((triple_encoding.shape[0], triple_encoding.shape[2]))
                         triple_encoding = triple_encoding.to(torch.float32)

                         time_encoding = time_encoding.reshape((time_encoding.shape[0], time_encoding.shape[2]))
                         time_encoding = time_encoding.to(torch.float32)

                         logits = model(triple_encoding, time_encoding)

                         if len(preds) == 0:
                             batch_logits = logits.detach().cpu().numpy()
                             preds.append(batch_logits)

                         else:
                             batch_logits = logits.detach().cpu().numpy()
                             preds[0] = np.append(preds[0], batch_logits, axis=0)

                 print(preds)

                 rel_values = np.array(preds)
                 rel_values = torch.tensor(rel_values)

                 print(rel_values, rel_values.shape)

                 #####
                 _, argsort1 = torch.sort(rel_values[0,:,0], descending=True)
                 rank_time_point = np.where(argsort1 == 0)[0][0]

                 print("The current time point is:" + str(time_point))
                 print("The rank for this time point is:" + str(rank_time_point))
                 rank_for_each_time_point.append(rank_time_point)

             print("rank_for_each_time_point:" + str(rank_for_each_time_point))
             rank2 = np.mean(np.array(rank_for_each_time_point))
             print("mean:" + str(rank2))
             ranks.append(rank2+1)
             ranks_right.append(rank2+1)
             if rank2 < 10:
                top_ten_hit_count += 1

             print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

             file_prefix = str(args.data_dir) + "_" + "nwords" + str(args.number_of_words) + "_"
             file_prefix += "tn" +str(args.n_temporal_neg) + "_" + "ctn" +str(args.n_corrupted_triple) + "_"
             file_prefix += "td" + str(args.time_dimension) + "_" + "ep" +str(args.epochs) + "_" + "lr" +str(args.lr) + "_" + "mr" + str(args.margin)

             f = open(file_prefix + '_intermediate_LP.txt','a')
             f.write(str(rank1) + '\t' + str(rank2) + '\n')
             f.close()

             # this could be done more elegantly, but here you go
             for hits_level in range(10):
                 if rank1 <= hits_level:
                     hits[hits_level].append(1.0)
                     hits_left[hits_level].append(1.0)
                 else:
                     hits[hits_level].append(0.0)
                     hits_left[hits_level].append(0.0)

                 if rank2 <= hits_level:
                     hits[hits_level].append(1.0)
                     hits_right[hits_level].append(1.0)
                 else:
                     hits[hits_level].append(0.0)
                     hits_right[hits_level].append(0.0)

         for i in [0,2,9]:
             print('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
             print('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
             print('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
         print('Mean rank left: {0}'.format(np.mean(ranks_left)))
         print('Mean rank right: {0}'.format(np.mean(ranks_right)))
         print('Mean rank: {0}'.format(np.mean(ranks)))
         print('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
         print('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
         print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

         file_prefix = str(args.data_dir) + "_" + "nwords" + str(args.number_of_words) + "_"
         file_prefix += "tn" +str(args.n_temporal_neg) + "_" + "ctn" +str(args.n_corrupted_triple) + "_"
         file_prefix += "td" + str(args.time_dimension) + "_" + "ep" +str(args.epochs) + "_" + "lr" +str(args.lr) + "_" + "mr" + str(args.margin)

         f = open(file_prefix + '_LP.txt','a')

         for i in [0,2,9]:
             f.write('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
             f.write('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
             f.write('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
         f.write('Mean rank left: {0}'.format(np.mean(ranks_left)))
         f.write('Mean rank right: {0}'.format(np.mean(ranks_right)))
         f.write('Mean rank: {0}'.format(np.mean(ranks)))
         f.write('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
         f.write('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
         f.write('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
         f.close()
if __name__ == "__main__":
    main()
