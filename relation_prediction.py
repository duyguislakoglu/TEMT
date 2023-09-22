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
import random
os.environ['CUDA_VISIBLE_DEVICES']= '0'

def main():
     # Training settings
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
     parser.add_argument("--cross_encoder",
                        action='store_true',
                        help="Whether to use whole sentence.")
     args = parser.parse_args()

     use_cuda = not args.no_cuda and torch.cuda.is_available()
     torch.manual_seed(args.seed)
     device = torch.device("cuda" if use_cuda else "cpu")

     if args.do_train:
         ################ M O D E L ######################

         processor = TKGProcessor(args.data_dir, "lp", "train", min_time=args.min_time, max_time=args.max_time, time_dimension=args.time_dimension)
         processor.n_temporal_neg = args.n_temporal_neg
         processor.n_corrupted_triple = args.n_corrupted_triple

         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

         def weights_init(m):
             if isinstance(m, nn.Linear):
                 #nn.init.xavier_uniform(m.weight)
                 torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                 torch.nn.init.zeros_(m.bias)
                 #nn.init.xavier_uniform(m.bias.data)

         def set_seed(seed: int = 42) -> None:
             np.random.seed(seed)
             random.seed(seed)
             torch.manual_seed(seed)
             torch.cuda.manual_seed(seed)
             # When running on the CuDNN backend, two further options must be set
             torch.backends.cudnn.deterministic = True
             torch.backends.cudnn.benchmark = False
             # Set a fixed value for the hash seed
             os.environ["PYTHONHASHSEED"] = str(seed)
             print(f"Random seed set as {seed}")

         #set_seed(0)

         model = CrossEncoderWithTime()
         #model.apply(weights_init)

         model.to(device)
         optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

         #loss_fn = nn.BCEsLoss()
         loss_fn = nn.MarginRankingLoss(margin=args.margin)

         model.train()


         for name, param in model.named_parameters():
             if param.requires_grad:
                 print(name)
                 print(param.data)

         ################ D A T A ##################

         #train_examples = processor.get_train_examples(args.data_dir)

         #train_features = processor.convert_examples_to_features(train_examples)
         data_prefix = str(args.data_dir.split("/")[-1]) + "_" + str(args.number_of_words) + "_"
         data_prefix += str(args.n_temporal_neg) + "_" + str(args.n_corrupted_triple) + "_"
         data_prefix += str(args.time_dimension)

         #with open("../"+str(data_prefix)+".dat", "wb") as f:
        #      pickle.dump(train_features, f)

         with open("../WIKIDATA12k_15_15_15_64ind.dat", "rb") as f:
             train_features = pickle.load(f)

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

                 if j == int(len(train_loader)/(args.n_temporal_neg + args.n_corrupted_triple +1)-2):
                     break

             losses.append(loss.item())
             print("epoch {}\tloss : {}".format(i,loss))

             if args.save_model:
                 torch.save(model.state_dict(), args.save_to)

     if args.do_test:

         time_vectors = processor.time_encoder.time_vectors
         processor = TKGProcessor(args.data_dir, task="lp", mode="test", min_time=args.min_time, max_time=args.max_time, time_dimension=args.time_dimension)
         processor.time_vectors = time_vectors
         
         model.eval()
         model.to(device)

         # run relation prediction
         ranks = []
         hits = []

         top_ten_hit_count = 0

         for i in range(10):
             hits.append([])

         print("Creating relations")
         with open(os.path.join(args.data_dir, "relation2id.txt"), 'r') as f:
             relation_ids = set()
             lines = f.readlines()
             for line in lines:
                 relation_ids.update([line.split("\t")[1][:-1]])
         relations = list(relation_ids)

         test_quadruples_lines = processor._read_tsv(os.path.join(args.data_dir, "test.txt"))
         d = 0
         for test_quadruple_line in test_quadruples_lines:
             rel_corrupt_lines = [test_quadruple_line]
             c = 0

             for tmp_rel in relations:
                 if tmp_rel != test_quadruple_line[1]:
                     tmp_triple = (test_quadruple_line[0], tmp_rel, test_quadruple_line[2])
                     if (test_quadruple_line[0], tmp_rel, test_quadruple_line[2]) not in processor.all_type_triples:
                         rel_corrupt_lines.append((test_quadruple_line[0], tmp_rel, test_quadruple_line[2] , test_quadruple_line[3], test_quadruple_line[4]))
                         #c += 1
                         #if c == 20:
                        #     break
                     else:
                         print("Cannot add this corrupted triple because it already exists in the graph.")
             print(rel_corrupt_lines)
             print(len(rel_corrupt_lines))

             tmp_examples = processor.create_examples(rel_corrupt_lines, args.data_dir, "only_ends")
             tmp_features = processor.convert_examples_to_features(tmp_examples)

             #with open("../tmp_features.dat", "wb") as f:
             #    pickle.dump(tmp_features, f)

             #with open("../tmp_features.dat", "rb") as f:
             #    tmp_features = pickle.load(f)

             all_time_encoding = torch.tensor([f.time_encoding for f in tmp_features])
             all_label = torch.tensor([f.label for f in tmp_features])
             all_triple_encoding = torch.tensor([f.triple_encoding for f in tmp_features])

             test_data = TensorDatasetWithMoreNegatives(all_time_encoding,
                        all_label, all_triple_encoding, number_of_negatives=processor.n_temporal_neg + processor.n_corrupted_triple)

             test_loader = DataLoader(test_data, batch_size=len(rel_corrupt_lines), shuffle=False)

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

             if (test_quadruple_line[3] == "#" or test_quadruple_line[4] == "#" or test_quadruple_line[3] == test_quadruple_line[4]):
                 _, argsort = torch.sort(rel_values[0,:,0], descending=True)
                 print(argsort)
                 #argsort1 = argsort1.cpu().numpy()
                 rank = np.where(argsort == 0)[0][0]
                 print('left: ', rank)

             else: # Which means it is closed and we need to choose the better one

                 even_rel_values = rel_values[0,:,0][::2]
                 odd_rel_values =  rel_values[0,:,0][1::2]

                 _, argsort1 = torch.sort(even_rel_values, descending=True)
                 rank1 = np.where(argsort1 == 0)[0][0]
                 print("Rank1 is " + str(rank1))

                 _, argsort2 = torch.sort(odd_rel_values, descending=True)
                 rank2 = np.where(argsort2 == 0)[0][0]
                 print("Rank2 is " + str(rank2))

                 # Choose which one is smaller
                 if rank1 < rank2:
                     rank = rank1
                 else:
                     rank = rank2

             ranks.append(rank+1)
             if rank < 10:
                 top_ten_hit_count += 1

             print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

             file_prefix = str(args.data_dir) + "_" + "nwords" + str(args.number_of_words) + "_"
             file_prefix += "tn" +str(args.n_temporal_neg) + "_" + "ctn" +str(args.n_corrupted_triple) + "_"
             file_prefix += "td" + str(args.time_dimension) + "_" + "ep" +str(args.epochs) + "_" + "lr" +str(args.lr) + "_" + "mr" + str(args.margin)

             f = open(file_prefix + '_intermediate_RP.txt','a')

             f.write(str(rank) + '\n')
             f.close()

             # this could be done more elegantly, but here you go
             for hits_level in range(10):
                 if rank <= hits_level:
                     hits[hits_level].append(1.0)
                 else:
                     hits[hits_level].append(0.0)

         for i in [0,2,9]:
             print('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
         print('Mean rank: {0}'.format(np.mean(ranks)))
         print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

         file_prefix = str(args.data_dir) + "_" + "nwords" + str(args.number_of_words) + "_"
         file_prefix += "tn" +str(args.n_temporal_neg) + "_" + "ctn" +str(args.n_corrupted_triple) + "_"
         file_prefix += "td" + str(args.time_dimension) + "_" + "ep" +str(args.epochs) + "_" + "lr" +str(args.lr) + "_" + "mr" + str(args.margin)

         f = open(file_prefix + '_RP.txt','a')

         for i in [0,2,9]:
             f.write('Hits @{0}: {1}'.format(i+1, np.mean(hits[i]))+'\n')
         f.write('Mean rank: {0}'.format(np.mean(ranks))+'\n')
         f.write('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks)))+ '\n')
         f.close()

if __name__ == "__main__":
    main()
