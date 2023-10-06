import pandas as pd
import numpy as np
import torch
import os
from models import CrossEncoderWithTime
from dataset import TensorDatasetWithMoreNegatives
import torch.nn as nn
from torch.nn import MarginRankingLoss
from eval.eval_time_prediction import eval_tp
import argparse
import pickle
from data_processor import TKGProcessor
from torch.utils.data import DataLoader, TensorDataset
import logging
import itertools

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
     parser.add_argument('--load_from', default='model.pth', type=str, help="Pretrained model checkpoint")


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

     parser.add_argument("--saved_test_features",
                        action='store_true',
                        help="Whether to use saved test features.")

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
     parser.add_argument('--constant_threshold', type=float, default=0.65, metavar='N',
                                   help='Score threshold for time prediction evaluation.')

     parser.add_argument("--sampling_type",
                        default="only_ends",
                        type=str)

     args = parser.parse_args()
     use_cuda = not args.no_cuda and torch.cuda.is_available()
     device = torch.device("cuda" if use_cuda else "cpu")
     print(device)

     if args.do_train:
         ################ M O D E L ######################
         processor = TKGProcessor(args.data_dir, "tp", "train", min_time=args.min_time, max_time=args.max_time, time_dimension=args.time_dimension)
         processor.n_temporal_neg = args.n_temporal_neg
         processor.n_corrupted_triple = args.n_corrupted_triple
         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         model = CrossEncoderWithTime(time_dimension=args.time_dimension)
         model.to(device)
         optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
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
             train_examples = processor.get_train_examples(args.data_dir, args.sampling_type)
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
             epoch_loss = []
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
                     epoch_loss.append(loss.item())
                     optimizer.zero_grad()
                     loss.backward()
                     optimizer.step()

             losses.append(sum(epoch_loss) / len(epoch_loss))
             print("epoch {}\tloss : {}".format(i,sum(epoch_loss) / len(epoch_loss)))
         if args.save_model:
            torch.save(model.state_dict(), args.save_to)

     if args.do_test:
         data_prefix = str(args.data_dir.split("/")[-1]) + "_" + str(args.number_of_words) + "_"
         data_prefix += str(args.n_temporal_neg) + "_" + str(args.n_corrupted_triple) + "_"
         data_prefix += str(args.time_dimension)
         processor = TKGProcessor(args.data_dir, task="tp", mode="test", min_time=args.min_time, max_time=args.max_time, time_dimension=args.time_dimension)
         time_vectors = processor.time_encoder.time_vectors
         ground_truth_intervals = processor.get_ground_truth_test_intervals(args.data_dir)

         if args.saved_test_features:
             with open("../test_{}.dat".format(data_prefix), "rb") as f:
                 test_features = pickle.load(f)
         else:
             test_examples = processor.get_test_examples(args.data_dir)
             test_features = processor.convert_examples_to_features(test_examples, use_descriptions=args.use_descriptions)
             with open("../test_{}.dat".format(data_prefix), "wb") as f:
                pickle.dump(test_features, f)

         all_time_encoding = torch.tensor([f.time_encoding for f in test_features])
         all_label = torch.tensor([f.label for f in test_features])
         all_triple_encoding = torch.tensor([f.triple_encoding for f in test_features])
         test_data = TensorDatasetWithMoreNegatives(all_time_encoding,
                    all_label, all_triple_encoding, number_of_negatives=0)
         batch_size = 1
         test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

         if args.pretrained:
             model = CrossEncoderWithTime(time_dimension=args.time_dimension)
             model.load_state_dict(torch.load(args.load_from))
             model.to(device)
         time_range = range(args.min_time, args.max_time+1)
         eval_tp(model, device, test_loader, time_vectors, time_range, ground_truth_intervals, args=args, constant_threshold=args.constant_threshold)

if __name__ == "__main__":
    main()
