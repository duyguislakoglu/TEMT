import os
import csv
import sys
import torch
import random
import logging
from sentence_transformers import SentenceTransformer
from time_encoder import PositionalEncoder
from collections import defaultdict
import nltk
import numpy as np

def get_first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return ""
    return sentences[0]

class QuadrupleExample(object):
    """A single training/test example."""

    def __init__(self, guid, subject_id=None, predicate_id=None, object_id=None,
                timestamp=None, label=None):
        """Constructs a Quadruple Example.

        Args:
            guid: Unique id for the example.
            subject_id: string.
            predicate_id: string.
            object_id: string.
            timestamp: string
            label: int. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.subject_id = subject_id
        self.predicate_id = predicate_id
        self.object_id = object_id
        self.timestamp = timestamp

        self.label = label

class QuadrupleFeatures(object):
    """A single set of features (vectors) of data."""

    def __init__(self, time_encoding, label, triple_encoding):
        self.time_encoding = time_encoding
        self.label =  label
        self.triple_encoding = triple_encoding # Needed for cross-encoder

class DataProcessor(object):
    """Base class for data converters for temporal knowledge graph data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class TKGProcessor(DataProcessor):
    """Processor for temporal knowledge graph data set."""
    def __init__(self, data_dir, task, mode, min_time=0, max_time=0,
    time_dimension=64, n_temporal_neg=0, n_corrupted_triple=0, number_of_words=15):
        self.task = task # tp, lp
        self.mode = mode # train, dev, test
        self.data_dir = data_dir
        self.create_triple_dictionary(data_dir, self.mode) # Creates triple:list of validity intervals dictionary
        self.number_of_words = number_of_words

        self.n_temporal_neg = n_temporal_neg # Number of temporal negatives
        self.n_corrupted_triple = n_corrupted_triple

        self.id2entity = self.create_inverse_dictionary(os.path.join(data_dir, "entity2id.txt"))
        self.id2predicate = self.create_inverse_dictionary(os.path.join(data_dir, "relation2id.txt"))

        self.entity2name = self.create_dictionary(os.path.join(data_dir, "entity2name.txt"))
        #print(self.entity2name)
        self.predicate2name = self.create_dictionary(os.path.join(data_dir, "relation2name.txt"))
        #print(self.predicate2name)
        self.entity2desc = self.create_dictionary(os.path.join(data_dir, "entity2desc.txt"))

        #self.predicate2desc = self.create_dictionary(os.path.join(data_dir, "relation2desc.txt"))
        self.min_time = min_time
        self.max_time = max_time
        self.time_dimension = time_dimension
        self.time_encoder = PositionalEncoder(self.min_time, self.max_time, self.time_dimension)
        #self.lm_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        #self.lm_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.lm_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        self.embedding_lookup = {} # For feature based approach
        self.triple_embedding_lookup = {}

        self.training_entities = self.get_training_entities(data_dir) # Needed for corrupting head or tail
        self.entities = self.get_entities(data_dir) # Needed for link prediction inference
        if task == "tp":
            self.all_times_list = range(self.min_time, self.max_time+1)

        if task == "lp":
            if mode == "test":
                self.all_type_triples = self.get_all_type_triples(data_dir)
                self.all_triple_dictionary = self.create_all_triple_dictionary(data_dir)

    def get_type_triples(self, data_dir, set_type): # Needed for checking negatives
        type_triples = set()
        path = os.path.join(data_dir, "{}.txt".format(set_type))

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                triple = (line[0],line[1],line[2])
                type_triples.add(triple)
        return type_triples

    def get_all_type_triples(self, data_dir): # Needed for checking negatives
        all_type_triples  = set()

        for set_type in ["train", "valid", "test"]:
            type_triples = self.get_type_triples(data_dir, set_type)
            all_type_triples.update(type_triples)
        return all_type_triples


    ########################

    def get_triple_dictionary(self, data_dir, mode):
        triple_dictionary  = {}
        path = os.path.join(data_dir, "{}.txt".format(mode))

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                triple = (line[0],line[1],line[2])
                interval = self.interval_from_text(line[3], line[4])
                if triple not in triple_dictionary.keys():
                    triple_dictionary[triple] = [interval]
                else:
                    triple_dictionary[triple].append(interval)
        return triple_dictionary

    # Needed for link prediction inference
    def create_all_triple_dictionary(self, data_dir):
        all_triple_dictionary  = defaultdict(list)

        tr_triples = self.get_triple_dictionary(data_dir, "train")

        vl_triples = self.get_triple_dictionary(data_dir, "valid")
        tst_triples = self.get_triple_dictionary(data_dir, "test")

        for d in (tr_triples, vl_triples, tst_triples): # you can list as many input dicts as you want here
            for key, value in d.items():
                all_triple_dictionary[key].append(value)

        all_triple_dictionary = {k:v[0] for k, v in all_triple_dictionary.items()}

        return all_triple_dictionary

    def in_tkg_lp(self, triple, time_point):
        if not (triple in self.all_triple_dictionary.keys()):
            return False
        if self.in_intervals(time_point, self.all_triple_dictionary[triple]):
            print(str(triple) + "already exist in the graph.")
            return True
        return False

    ########################
    def create_inverse_dictionary(self, path):
        dict = {}
        with open(path, 'r') as f:
            for line in f:
               (v, k) = line.split('\t')[0:2]
               if k[-1] == "\n":
                   dict[k[:-1]] = v
               else:
                   dict[k] = v
        return dict

    def create_dictionary(self, path): # Needed for descriptions
        dict = {}
        with open(path, 'r') as f:
            for line in f:
                if len(line) == 1:
                    break
                (k, v) = line.split('\t')[0:2]
                if "name" in path:
                    #if "Wikidata12k" in path:
                    #    dict[k] = v[:-1]
                    #else:
                    #dict[k] = v
                    dict[k] = v[:-1]
                else:
                    dict[k] = v
                    if dict[k][-1] == '\n':
                        dict[k] = dict[k][:-1]
        return dict

    def get_labels(self, data_dir): # not needed for now
        """Gets all labels (0, 1) for quadruples in the temporal knowledge graph."""
        return [0, 1]

    def get_train_examples(self, data_dir, sampling_type="only_ends"):
        """See base class."""
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "train.txt")), data_dir, sampling_type)

    def get_dev_examples(self, data_dir, sampling_type="only_ends"):
        """See base class."""
        if self.task == "tp":
            return self.create_examples(self._read_tsv(os.path.join(data_dir, "intervals/valid.txt")), data_dir, sampling_type)
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "valid.txt")), data_dir, sampling_type)

    def get_test_examples(self, data_dir, sampling_type="only_ends"):
      """See base class."""
      if self.task == "tp":
          return self.create_examples(self._read_tsv(os.path.join(data_dir, "intervals/test.txt")), data_dir, sampling_type)
      return self.create_examples(self._read_tsv(os.path.join(data_dir, "test.txt")), data_dir, sampling_type)

    def get_training_entities(self, data_dir):
        with open(os.path.join(data_dir, "train.txt"), 'r') as f:
            training_entities = set()
            lines = f.readlines()
            for line in lines:
                training_entities.update([line.split("\t")[0] , line.split("\t")[2]])
        return list(training_entities)

    def get_entities(self, data_dir):
        with open(os.path.join(data_dir, "entity2id.txt"), 'r') as f:
            entities = set()
            lines = f.readlines()
            for line in lines:
                if "WIKIDATA" in data_dir:
                    entities.update([line.split("\t")[1][:-1]])
                else:
                    entities.update([line.split("\t")[1]])
        return list(entities)

    # TODO: send it to helper
    def interval_from_text(self, s_text, e_text):
        if s_text[0] != "-":
            s_text = s_text.split("-")[0]
        else:
            s_text = "-" + s_text.split("-")[1]

        if e_text[0] != "-":
            e_text = e_text.split("-")[0]
        else:
            e_text = "-" + e_text.split("-")[1]

        lst = []
        if not s_text[0] == "#":
            lst.append(int(s_text.replace("#", "0")))
        if not e_text[0] == "#" and (s_text!=e_text):
            lst.append(int(e_text.replace("#", "0")))
        return lst

    # Creates triple dictionary with (s,p,o): validity_intervals
    def create_triple_dictionary(self, data_dir, mode):
        self.triple_dictionary  = {}
        path = os.path.join(data_dir, "{}.txt".format(mode))

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                triple = (line[0],line[1],line[2])
                interval = self.interval_from_text(line[3], line[4])
                if triple not in self.triple_dictionary.keys():
                    self.triple_dictionary[triple] = [interval]
                else:
                    self.triple_dictionary[triple].append(interval)

    # parameter lst_of_intervals is list of lists in the following form: [start,end]
    def in_intervals(self, neg_candidate_year, lst_of_intervals):
        for interval in lst_of_intervals:
            # TODO: Change this with Date class
            if len(interval) == 1: # Checking a timestamp
                if neg_candidate_year == interval[0]:
                    return True
            elif neg_candidate_year in range(interval[0], interval[1] +1):
                return True
        return False

    def in_tkg(self, triple, neg_candidate_time):
        # We know that triple is already correct. So we are checking the intervals only.
        if self.in_intervals(neg_candidate_time, self.triple_dictionary[triple]):
            return True
        return False

    def in_kg(self, triple):
        if triple in self.triple_dictionary.keys():
            return True
        return False

    def get_ground_truth_test_intervals(self, data_dir):

        ground_truth_intervals = []

        path = os.path.join(data_dir, "intervals/test.txt")

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\t")
                text_start_time = line[3]
                text_end_time = line[4]
                start_time = text_start_time.split("-")[0]
                end_time = text_end_time.split("-")[0]

                # Faulty example where end time is earlier then start time.
                if text_start_time[0] != "#" and text_end_time[0] != "#" and text_start_time != text_end_time:
                    s = int(text_start_time.split("-")[0].replace("#", "0"))
                    e = int(text_end_time.split("-")[0].replace("#", "0"))
                    if s > e:
                        print("Skipping faulty example in ground_truth_intervals")
                        print(line)
                        continue

                if start_time[0] == "#": # Left open
                    ground_truth_intervals.append((end_time, end_time))

                elif end_time[0] == "#": # Right open
                    ground_truth_intervals.append((start_time, start_time))

                else:
                    ground_truth_intervals.append((start_time, end_time))

        return ground_truth_intervals

    def get_ground_truth_val_intervals(self, data_dir):

        ground_truth_intervals = []

        path = os.path.join(data_dir, "intervals/valid.txt")

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\t")
                text_start_time = line[3]
                text_end_time = line[4]
                start_time = text_start_time.split("-")[0]
                end_time = text_end_time.split("-")[0]

                if start_time[0] == "#": # Left open
                    ground_truth_intervals.append((end_time, end_time))

                elif end_time[0] == "#": # Right open
                    ground_truth_intervals.append((start_time, start_time))

                else:
                    ground_truth_intervals.append((start_time, end_time))

        return ground_truth_intervals



    def get_temporal_negatives(self, triple, t, n, time_type, guid_number_prefix):
        """
            Creates n time dependent negatives.
        """
        if n == 0:
            return []
        # Get valid intervals for the triple
        valid_intervals = self.triple_dictionary[triple]
        negative_list = []

        for i in range(n):
            counter = 0
            while True:
                # Get candidate
                neg_candidate_time = random.choice(self.all_times_list)
                counter += 1

                if time_type == "left_open":
                    if neg_candidate_time > t and (not self.in_tkg(triple, neg_candidate_time)):
                        break
                elif time_type == "right_open":
                    if neg_candidate_time < t and (not self.in_tkg(triple, neg_candidate_time)):
                        break
                else:
                    if not self.in_tkg(triple, neg_candidate_time):
                         break

            # Create quadruple and add it to list
            print("Negative: " +  triple[0] + " " + triple[1] + " " +  triple[2] + " " + str(neg_candidate_time))
            negative_list.append(QuadrupleExample(guid_number_prefix+"_"+str(i), triple[0], triple[1], triple[2], neg_candidate_time, label=0))

        # Return list of negative Quadruples
        return negative_list

    def corrupt_head(self, triple, timestamp, guid):
        tmp_ent_list = self.training_entities

        while True:
            tmp_head = random.choice(tmp_ent_list)
            if (tmp_head, triple[1], triple[2]) not in self.triple_dictionary:
                break
            print("Found {} in triple dictionary so trying again to corrupt.".format((tmp_head, triple[1], triple[2])))
        print("Corrupted head...")
        print(str(tmp_head) + " " + str(triple[1]) + " " + str(triple[2]) + " " + str(timestamp))
        return QuadrupleExample(guid, tmp_head, triple[1], triple[2], timestamp, 0)

    def corrupt_tail(self, triple, timestamp, guid):
        tmp_ent_list = self.training_entities

        while True:
            tmp_tail = random.choice(tmp_ent_list)
            if (triple[0], triple[1], tmp_tail) not in self.triple_dictionary:
                break
            print("Found {} in triple dictionary so trying again to corrupt.".format((triple[1], triple[1], tmp_tail)))

        print("Corrupted tail...")
        print(str(triple[0]) + " " + str(triple[1]) + " " + str(tmp_tail) + " " + str(timestamp))
        return QuadrupleExample(guid, triple[0], triple[1], tmp_tail, timestamp, 0)

    def corrupt_triple(self, triple, timestamp, n, guid_number_prefix):
        """
            Creates n time independent negatives.
        """
        if n == 0:
            return []
        if self.task == "tc" and self.mode == "test":
            self.triple_dictionary = self.create_all_triple_dictionary(self.data_dir)

        negative_list = []
        for i in range(n):
            rnd = random.random()
            if rnd <= 0.5: # Corrupting the head
                negative_list.append(self.corrupt_head(triple, timestamp, guid_number_prefix+"_"+str(i)))
            else:
                negative_list.append(self.corrupt_tail(triple, timestamp, guid_number_prefix+"_"+str(i)))
        return negative_list

    def get_timestamps(self, text_start_time, text_end_time, sampling_type, number_of_samples):
        # This is dataset-specific

        # TODO: This comes from year granularity. Make it generic

        if text_start_time[0] != "-":
            start_time = text_start_time.split("-")[0]
        else:
            start_time = "-" + text_start_time.split("-")[1]

        if text_end_time[0] != "-":
            end_time = text_end_time.split("-")[0]
        else:
            end_time = "-" + text_end_time.split("-")[1]

        time_type = ""
        lst = []

        if start_time[0] == "#": # Left open
            time_type = "left_open"
            lst.append(int(end_time.replace("#", "0")))

        elif end_time[0] == "#": # Right open
            time_type = "right_open"
            lst.append(int(start_time.replace("#", "0")))

        elif start_time == end_time: # Timestamp
            time_type = "timestamp"
            lst.append(int(start_time.replace("#", "0")))

        else:
            time_type = "closed"
            if sampling_type == "only_ends":
                lst = [int(start_time.replace("#", "0")), int(end_time.replace("#", "0"))]
            if sampling_type == "all":
                lst = list(range(int(start_time.replace("#", "0")), int(end_time.replace("#", "0"))+1))
            # TODO: number_of_samples will be used when we sample from the intermediate years
        return (lst, time_type)

    def get_entity_text(self, entity_id):
        entity = self.id2entity[entity_id]
        return self.entity2name[entity]

    def get_entity_desc(self, entity_id):
        entity = self.id2entity[entity_id]
        first_sentence = get_first_sentence(self.entity2desc[entity])
        if "is a stub." in first_sentence:
            clean_first_sentence = first_sentence.split(".")[:-2]
            return "".join(clean_first_sentence)+"."
        return first_sentence

    def get_predicate_text(self, predicate_id):
        predicate = self.id2predicate[predicate_id]
        return self.predicate2name[predicate]

        #return self.predicate2name[predicate][:-1] + " " + self.predicate2desc[predicate]

    def get_entity_id(self, entity):
        return self.entity2id[entity]

    def get_predicate_id(self, predicate):
        return self.predicate2id[entity]

    # n_temporal_neg and n_corrupted_triple depend on the task
    def create_examples(self, lines, data_dir, sampling_type):
        examples = []

        for (i, line) in enumerate(lines):
            subject_id = line[0]
            predicate_id = line[1]
            object_id = line[2]
            text_start_time = line[3]
            text_end_time = line[4]

            # Faulty example where end time is earlier then start time.
            if text_start_time[0] != "#" and text_end_time[0] != "#" and text_start_time != text_end_time:
                s = int(text_start_time.split("-")[0].replace("#", "0"))
                e = int(text_end_time.split("-")[0].replace("#", "0"))
                if s > e:
                    print("Skipping faulty example in the line")
                    print(line)
                    continue

            print("################## Line")
            print(line)

            guid = "%s-%s" % (self.mode, i)

            # number_of_samples will be used when we sample from the intermediate years
            number_of_samples = 0
            (timestamps, time_type) = self.get_timestamps(text_start_time, text_end_time, sampling_type, number_of_samples)
            print(time_type)

            if self.task == "tp" and self.mode == "test" and time_type=="closed": # No need to create 2 examples for test
                examples.append(
                    QuadrupleExample(guid, subject_id, predicate_id, object_id, timestamp=timestamps[0], label=1))
                print("###### Example")
                print(subject_id + " " + predicate_id + " " +  object_id + " " + str(timestamps[0]))

            elif self.task == "tc":
                if self.mode == "test":
                    # Check whether this test candidate triple is in train or val set
                    if ((subject_id, predicate_id, object_id) in self.get_triple_dictionary(data_dir, "train")) or ((subject_id, predicate_id, object_id) in self.get_triple_dictionary(data_dir, "valid")) :
                        print(subject_id + " " + predicate_id + " " + object_id + " is in training set. Filtering out.")
                        continue

                examples.append(
                    QuadrupleExample(guid, subject_id, predicate_id, object_id, timestamp=timestamps[0], label=1))
                print("###### Example")
                print(subject_id + " " + predicate_id + " " +  object_id + " " + str(timestamps[0]))

                #temporal_negatives = self.get_temporal_negatives((subject_id, predicate_id, object_id), timestamps[0], self.n_temporal_neg, time_type, guid)
                #examples = examples + temporal_negatives
                corrupted_triples = self.corrupt_triple((subject_id, predicate_id, object_id), timestamps[0], self.n_corrupted_triple, guid)
                examples = examples + corrupted_triples

            else:
                for timestamp in timestamps:
                    examples.append(
                        QuadrupleExample(guid, subject_id, predicate_id, object_id, timestamp=timestamp, label=1))
                    print("###### Example")
                    print(subject_id + " " + predicate_id + " " +  object_id + " " + str(timestamp))
                    # Get negatives
                    if self.mode == "train":
                        temporal_negatives = self.get_temporal_negatives((subject_id, predicate_id, object_id), timestamp, self.n_temporal_neg, time_type, guid)
                        examples = examples + temporal_negatives
                        corrupted_triples = self.corrupt_triple((subject_id, predicate_id, object_id), timestamp, self.n_corrupted_triple, guid)
                        examples = examples + corrupted_triples
                    print("######")
                print("######################")

        if self.task == "tc":
            self.write_to_csv(examples, self.mode)

        return examples

    def write_to_csv(self, examples, mode):
        # open the file in the write mode
        with open('tc_{}.csv'.format(mode), 'w', encoding='UTF8') as f:
        # create the csv writer
            writer = csv.writer(f)
            for (ex_index, example) in enumerate(examples):
                row = str(example.subject_id) +"\t"+ str(example.predicate_id) +"\t"+ str(example.object_id) +"\t"+ str(example.label)
                writer.writerow([row])

    #def truncate(subject_text, predicate_text, object_text))
    #    total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)

    def get_textual_features(self, examples, use_descriptions=False):
        textual_features = []
        for (ex_index, example) in enumerate(examples):

            triple_embedding_lookup = {}

            subject_text = self.get_entity_text(example.subject_id)
            predicate_text = self.get_predicate_text(example.predicate_id)
            object_text = self.get_entity_text(example.object_id)

            triple_text = subject_text +" "+ predicate_text +" "+ object_text+ "."

            if use_descriptions:
                descs = " ("

                subj_desc = self.get_entity_desc(example.subject_id)
                obj_desc = self.get_entity_desc(example.object_id)

                if len(subj_desc) > 2:
                    descs += subj_desc
                if len(obj_desc) > 2:
                    descs += " " + obj_desc
                descs += ")"

                triple_text += descs

            if (example.subject_id, example.predicate_id, example.object_id) not in triple_embedding_lookup:
                textual_feature = self.lm_encoder.encode(triple_text)
                triple_embedding_lookup[(example.subject_id, example.predicate_id, example.object_id)] = textual_feature

            else:
                textual_feature = triple_embedding_lookup[(example.subject_id, example.predicate_id, example.object_id)]
            textual_feature = np.append(textual_feature, int(example.label))
            textual_features.append(textual_feature)

        return textual_features

    def convert_examples_to_features(self, examples, use_descriptions=False):
        features = []
        for (ex_index, example) in enumerate(examples):

            subject_text = self.get_entity_text(example.subject_id)

            predicate_text = self.get_predicate_text(example.predicate_id)
            object_text = self.get_entity_text(example.object_id)
            time_encoding = self.time_encoder(example.timestamp)

            #triple_text = truncate(subject_text, predicate_text, object_text)
            triple_text = subject_text +" "+ predicate_text +" "+ object_text+ "."

            if use_descriptions:
                descs = " ("

                subj_desc = self.get_entity_desc(example.subject_id)
                obj_desc = self.get_entity_desc(example.object_id)

                if len(subj_desc) > 2:
                    descs += subj_desc
                if len(obj_desc) > 2:
                    descs += " " + obj_desc
                descs += ")"

                triple_text += descs

            print(triple_text)

            if (example.subject_id, example.predicate_id, example.object_id) not in self.triple_embedding_lookup:
                triple_encoding = self.lm_encoder.encode(triple_text)

                self.triple_embedding_lookup[(example.subject_id, example.predicate_id, example.object_id)] = triple_encoding
            else:
                triple_encoding = self.triple_embedding_lookup[(example.subject_id, example.predicate_id, example.object_id)]

            quad_features = QuadrupleFeatures(time_encoding, example.label, triple_encoding)

            features.append(quad_features)

        return features
