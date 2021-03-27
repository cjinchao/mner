import os
import random
import logging
import argparse

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from modeling.tokenization import BertTokenizer
from modeling.bert import MTCCMBertForMMTokenClassificationCRF

logger = logging.getLogger(__name__)

class MNERInputExample(object):

    def __init__(self, guid, text, image_id, label=None, aux_label=None):
        self.guid = guid
        self.text = text
        self.image_id = image_id
        self.label = label
        self.aux_label = aux_label

class MNERInputFeatures(object):

    def __init__(self, input_ids, input_mask, added_input_mask, token_type_ids, image_feature, label_id, aux_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.token_type_ids = token_type_ids
        self.image_feature = image_feature
        self.label_id = label_id
        self.aux_label_id = aux_label_id


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser("Multi-modal NER parser")
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    args = parser.parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        num_gpus = 1
        torch.distributed.init_process_group(backend="nccl")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid parameter gradient_accumuation_steps: {}, should be an integer >= 1".format(args.gradient_accumuation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumuation_steps
    set_random_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of do_train and do_eval must be true")

    processor = MNERProcessor()
    label_list = processor.get_label_list()
    aux_label_list = processor.get_label_list()

    trans_matrix = np.zeros((len(aux_label_list)+1, len(label_list)+1), dtype=float)
    trans_matrix[0,0]=1 # pad to pad
    trans_matrix[1,1]=1 # O to O
    trans_matrix[2,2]=0.25 # B to B-MISC
    trans_matrix[2,4]=0.25 # B to B-PER
    trans_matrix[2,6]=0.25 # B to B-ORG
    trans_matrix[2,8]=0.25 # B to B-LOC
    trans_matrix[3,3]=0.25 # I to I-MISC
    trans_matrix[3,5]=0.25 # I to I-PER
    trans_matrix[3,7]=0.25 # I to I-ORG
    trans_matrix[3,9]=0.25 # I to I-LOC
    trans_matrix[4,10]=1   # X to X
    trans_matrix[5,11]=1   # [CLS] to [CLS]
    trans_matrix[6,12]=1   # [SEP] to [SEP]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    train_examples = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
    
    mner_model = MTCCMBertForMMTokenClassificationCRF.from_pretrained(args.bert_model)
    image_encoder = getattr(resnet, 'resnet152')
    image_encoder.load_state_dict('resnet152.pth')
    image_encoder = myResnet(image_encoder, True, device)

    if args.fp16:
        image_encoder = image_encoder.half()
        mner_model = mner_model.half()

    image_encoder.to(device)
    mner_model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistibutedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    else:
        if num_gpus > 1:
            mner_model = torch.nn.DataParallel(mner_model)
            image_encoder = torch.nn.DataParallel(image_encoder)
    
    param_optimizer = list(mner_model.parameters())
    no_decay = ["bias", 'LayerNorm.bias', "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer, FusedAdam
        except:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, max_grad_norm=1.0)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    train_loss = 0
    global_step = 0

    if args.do_train:
        train_features = convert_examples_to_features(train_examples)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_feature], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_image_feats = torch.stack([f.image_feats for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_aux_label_ids = torch.tensor([f.aux_label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_image_feats, all_label_ids, all_aux_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
    logger.info("***** Run training *****")
    for idx in range(int(args.num_train_epochs)):
        logger.info("Epoch: " + str(idx) + '*' * 10)
        logger.info('Number of examples: {}'.format(len(train_examples)))
        mner_model.train()
        image_encoder.train()
        image_encoder.zero_grad()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, added_input_mask, segmemt_ids, image_feats, label_ids, aux_label_ids = batch
            with torch.no_grad():
                image_f, image_mean, image_att = image_encoder(image_feats)
            trans_matrix = torch.tensor(trans_matrix).to(device)
            loss = mner_model(input_ids, segment_ids, input_mask, added_input_mask, image_att, trans_matrix, label_ids, aux_label_ids)

            if num_gpus > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

