import math
import os
import json
import argparse
from retriever import DualEncoder
from datetime import datetime
from collections import OrderedDict
from utils import Logger
from transformers import BertTokenizer, BertModel, ElectraModel, ElectraTokenizer
from reader import Reader, get_predicts, prune_predicts
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn as nn
import torch
import numpy as np
import faiss


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_data(data_dir):
    samples = []
    with open(os.path.join(data_dir, "retriever_result.jsonl")) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def load_entities(kb_dir):
    entities = []
    with open(os.path.join(kb_dir, "entities_kilt.json")) as f:
        for line in f:
            entities.append(json.loads(line))
    return entities


def get_encoder(type_encoder, return_tokenizer=False):
    if type_encoder == "bert_base":
        encoder = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif type_encoder == "bert_large":
        encoder = BertModel.from_pretrained("bert-large-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    elif type_encoder == "electra_base":
        encoder = ElectraModel.from_pretrained("google/electra-base-discriminator")
        tokenizer = ElectraTokenizer.from_pretrained(
            "google/electra-base-discriminator"
        )
    elif type_encoder == "electra_large":
        encoder = ElectraModel.from_pretrained("google/electra-large-discriminator")
        tokenizer = ElectraTokenizer.from_pretrained(
            "google/electra-large-discriminator"
        )
    elif type_encoder == "squad2_bert_large":
        encoder = BertModel.from_pretrained("phiyodr/bert-large-finetuned-squad2")
        tokenizer = BertTokenizer.from_pretrained("phiyodr/bert-large-finetuned-squad2")
    elif type_encoder == "squad2_electra_large":
        encoder = ElectraModel.from_pretrained(
            "ahotrod/electra_large_discriminator_squad2_512"
        )
        tokenizer = ElectraTokenizer.from_pretrained(
            "ahotrod/electra_large_discriminator_squad2_512"
        )
    else:
        raise ValueError("wrong encoder type")
    if return_tokenizer:
        return encoder, tokenizer
    else:
        return encoder


def load_model(
    model_path,
    type_encoder,
    device,
    type_span_loss,
    do_rerank,
    type_rank_loss,
    max_answer_len,
    max_passage_len,
):
    encoder, tokenizer = get_encoder(type_encoder, True)
    package = (
        torch.load(model_path)
        if device.type == "cuda"
        else torch.load(model_path, map_location=torch.device("cpu"))
    )
    model = Reader(
        encoder,
        type_span_loss,
        do_rerank,
        type_rank_loss,
        max_answer_len,
        max_passage_len,
    )
    try:
        model.load_state_dict(package["sd"])
    except RuntimeError:
        # forgot to save model.module.sate_dict
        from collections import OrderedDict

        state_dict = package["sd"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            # for loading our old version reader model
            if name != "topic_query":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model, tokenizer


class ReaderData(Dataset):
    # get the input data item for the reader model
    def __init__(
        self,
        tokenizer,
        samples,
        entities,
        max_len,
        max_num_candidates,
        is_training,
        use_title=False,
    ):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.samples = samples
        self.entities = entities
        self.all_entity_token_ids = np.array([e["text_ids"] for e in entities])
        self.all_entity_masks = np.array([e["text_masks"] for e in entities])
        self.max_len = max_len
        self.max_num_candidates = max_num_candidates
        self.use_title = use_title
        self.TT = [2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        title = None
        if self.use_title:
            title = sample["title_ids"]
        token_ids = sample["token_ids"]
        passage_labels = sample["passage_labels"][: self.max_num_candidates]
        if self.use_title:
            title_ids = self.TT + title
        else:
            title_ids = []

        candidates = sample["candidates"][: self.max_num_candidates]
        candidates_ids = self.all_entity_token_ids[candidates]
        candidates_masks = self.all_entity_masks[candidates]

        encoded_pairs = torch.zeros((self.max_num_candidates, self.max_len)).long()
        type_marks = torch.zeros((self.max_num_candidates, self.max_len)).long()
        attention_masks = torch.zeros((self.max_num_candidates, self.max_len)).long()
        answer_masks = torch.zeros((self.max_num_candidates, self.max_len)).long()
        passage_labels = torch.tensor(passage_labels).long()
        for i, candidate_ids in enumerate(candidates_ids):
            candidate_ids = candidate_ids.tolist()
            candidate_masks = candidates_masks[i].tolist()
            # CLS mention ids TT title ids SEP candidate ids SEP
            input_ids = (
                token_ids[:-1]
                + title_ids
                + [self.tokenizer.sep_token_id]
                + candidate_ids[1:]
            )
            input_ids = (
                input_ids
                + [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            )[: self.max_len]
            attention_mask = [1] * (len(token_ids + title_ids)) + candidate_masks[1:]
            attention_mask = (
                attention_mask + [0] * (self.max_len - len(attention_mask))
            )[: self.max_len]
            token_type_ids = [0] * len(token_ids + title_ids) + candidate_masks[1:]
            token_type_ids = (
                token_type_ids + [0] * (self.max_len - len(token_type_ids))
            )[: self.max_len]
            encoded_pairs[i] = torch.tensor(input_ids)
            attention_masks[i] = torch.tensor(attention_mask)
            type_marks[i] = torch.tensor(token_type_ids)
            answer_masks[i, : len(token_ids)] = 1

        return (
            encoded_pairs,
            attention_masks,
            type_marks,
            answer_masks,
            passage_labels,
        )


def make_single_loader(data_set, bsz, shuffle):
    loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader


def get_loaders(
    tokenizer,
    samples,
    entities,
    max_len,
    max_num_candidates,
    bsz,
    use_title,
):
    samples_set = ReaderData(
        tokenizer,
        samples,
        entities,
        max_len,
        max_num_candidates,
        False,
        use_title,
    )
    loader = make_single_loader(samples_set, bsz, False)
    return loader


def get_raw_results(
    model, device, loader, k, samples, do_rerank, filter_span=True, no_multi_ents=False
):
    model.eval()
    ranking_scores = []
    ranking_labels = []
    ps = []
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            if do_rerank:
                batch_p, rank_logits_b = model(*batch)
            else:
                batch_p = model(*batch).detach()
            batch_p = batch_p.cpu()
            ps.append(batch_p)
            if do_rerank:
                ranking_scores.append(rank_logits_b.cpu())
                ranking_labels.append(batch[4].cpu())
        ps = torch.cat(ps, 0)
    raw_predicts = get_predicts(ps, k, filter_span, no_multi_ents)
    assert len(raw_predicts) == len(samples)
    if do_rerank:
        ranking_scores = torch.cat(ranking_scores, 0)
        ranking_labels = torch.cat(ranking_labels, 0)
    else:
        ranking_scores = None
        ranking_labels = None
    return raw_predicts, ranking_scores, ranking_labels


def transform_predicts(preds, entities, samples):
    #  ent_idx,start,end --> start, end, ent name
    ent_titles = [e["title"] for e in entities]
    assert len(preds) == len(samples)
    results = []
    for ps, s in zip(preds, samples):
        results_p = []
        for p in ps:
            ent_title = ent_titles[s["candidates"][p[0]]]
            r = p[1:]
            # start, end, entity name
            r.append(ent_title)
            results_p.append(r)
        results.append(results_p)
    return results


# save passage level results
def save_results(predicts, samples, results_dir):
    save_path = os.path.join(results_dir, "reader_results.jsonl")
    results = []
    for predict, sample in zip(predicts, samples):
        result = {}
        result["doc_id"] = sample["doc_id"]
        result["text"] = sample["token_text"]
        result["predicts"] = predict
        results.append(result)
    with open(save_path, "w") as f:
        for r in results:
            f.write("%s\n" % json.dumps(r))


def main(args):
    set_seeds(args)
    best_val_perf = float("-inf")
    logger = Logger(args.model + ".log", on=True)
    logger.log(str(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {str(device)}", force=True)
    start_time = datetime.now()
    # configure logger
    args.device = device
    samples = load_data(args.data_dir)
    entities = load_entities(args.kb_dir)
    model, tokenizer = load_model(
        args.model,
        args.type_encoder,
        device,
        args.type_span_loss,
        args.do_rerank,
        args.type_rank_loss,
        args.max_answer_len,
        args.max_passage_len,
    )

    loader = get_loaders(
        tokenizer,
        samples,
        entities,
        args.L,
        args.C,
        args.B,
        args.use_title,
    )
    model.to(device)
    args.n_gpu = torch.cuda.device_count()
    dp = args.n_gpu > 1
    if dp:
        logger.log(
            "Data parallel across {:d} GPUs {:s}"
            "".format(len(args.gpus.split(",")), args.gpus)
        )
        model = nn.DataParallel(model)
    model.eval()
    logger.log("getting test raw predicts")
    start_time_test_infer = datetime.now()
    raw_predicts = get_raw_results(
        model,
        device,
        loader,
        args.k,
        entities,
        args.do_rerank,
        args.filter_span,
        args.no_multi_ents,
    )
    pruned_preds = prune_predicts(raw_predicts, args.thresd)
    predicts = transform_predicts(pruned_preds, entities, samples)
    save_results(predicts, samples, args.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--data_dir", type=str, help=" data directory")
    parser.add_argument("--kb_dir", type=str, help=" kb directory")
    parser.add_argument("--results_dir", type=str, help=" results directory")
    parser.add_argument(
        "--type_encoder", type=str, default="bert_base", help="the type of encoder"
    )
    parser.add_argument(
        "--type_span_loss",
        type=str,
        default="sum_log",
        choices=["log_sum", "sum_log", "sum_log_nce", "max_min"],
        help="type of multi-label loss for span ?",
    )
    parser.add_argument(
        "--do_rerank", action="store_true", help="do rerank multi-tasking?"
    )
    parser.add_argument(
        "--type_rank_loss",
        type=str,
        default="sum_log",
        choices=["log_sum", "sum_log", "sum_log_nce", "max_min"],
        help="type of multi-label loss  for rerank?",
    )
    parser.add_argument(
        "--max_answer_len",
        type=int,
        default=10,
        help="max length of answer [%(default)d]",
    )
    parser.add_argument(
        "--max_passage_len",
        type=int,
        default=32,
        help="max length of passage [%(default)d]",
    )
    parser.add_argument("--filter_span", action="store_true", help="filter span?")
    parser.add_argument(
        "--no_multi_ents",
        action="store_true",
        help="prevent multiple entities for a mention span?",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="get top-k spans per entity before top-p " "filtering",
    )
    parser.add_argument(
        "--use_title", action="store_true", help="use title or use topic?"
    )
    parser.add_argument(
        "--C", type=int, default=64, help="max number of candidates [%(default)d]"
    )
    parser.add_argument(
        "--L", type=int, default=160, help="max length of joint input [%(default)d]"
    )
    parser.add_argument("--B", type=int, default=16, help="batch size [%(default)d]")
    parser.add_argument(
        "--thresd",
        type=float,
        default=0.05,
        help="probabilty threshold for top-p filtering",
    )
    args = parser.parse_args()

    # Set environment variables before all else.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # Sets torch.cuda behavior

    main(args)

# python ./simple_reader.py  --model ./models/reader.pt   --data_dir ./reader_results/  --C 100  --B 32  --L 180  --gpus 0  --lr 1e-5 --thresd  0.05  --k 3  --stride 16 --max_passage_len 32  --filter_span  --type_encoder squad2_electra_large  --type_span_loss sum_log  --type_rank_loss sum_log  --do_rerank  --use_title  --results_dir ./reader_results/  --kb_dir ./models/data/kb/
