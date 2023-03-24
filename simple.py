import math
import os
import json
import argparse
from retriever import DualEncoder
from datetime import datetime
from collections import OrderedDict
from utils import Logger
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn as nn
import torch
import numpy as np
import faiss


def load_data(data_dir):
    samples = []
    with open(os.path.join(data_dir, "samples.jsonl")) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def tokenize_original_text(processed_raw_data, tokenizer, args):
    data = []
    for d in processed_raw_data:
        orig_text = d["text"]
        orig_title = d["title"]
        text = tokenizer.tokenize(orig_text)
        doc_id = d["doc_id"]
        text_ids = tokenizer.convert_tokens_to_ids(text)
        title_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(orig_title))

        content_length = args["instance_length"] - 2

        if len(text_ids) < content_length:
            text_ids = [101] + tokenizer.convert_tokens_to_ids(text) + [102]
            data.append(
                {
                    "doc_id": doc_id,
                    "text": text_ids,
                    "title": title_ids,
                    "offset": 0,
                }
            )
        else:
            # -2 for [BOS] and [EOS]
            for ins_num in range(math.ceil(len(text_ids) / args["stride"])):
                begin = ins_num * args["stride"]
                end = ins_num * args["stride"] + content_length
                instance_ids = [101] + text_ids[begin:end] + [102]
                data.append(
                    {
                        "doc_id": doc_id,
                        "text": instance_ids,
                        "title": title_ids,
                        "offset": begin,
                    }
                )
    return data


def load_entities(kb_dir):
    entities = []
    with open(os.path.join(kb_dir, "entities_kilt.json")) as f:
        for line in f:
            entities.append(json.loads(line))

    return entities


def get_entity_map(entities):
    #  get all entity map: map from entity title to index
    entity_map = {}
    for i, e in enumerate(entities):
        entity_map[e["title"]] = i
    assert len(entity_map) == len(entities)
    return entity_map


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_model(config_path, model_path, device, type_loss, blink=True):
    with open(config_path) as json_file:
        params = json.load(json_file)
    if blink:
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
    else:
        ctxt_bert = BertModel.from_pretrained("bert-large-uncased")
        cand_bert = BertModel.from_pretrained("bert-large-uncased")
    state_dict = (
        torch.load(model_path)
        if device.type == "cuda"
        else torch.load(model_path, map_location=torch.device("cpu"))
    )
    model = DualEncoder(ctxt_bert, cand_bert, type_loss)
    model.load_state_dict(state_dict["sd"])
    return model


class EntitySet(Dataset):
    def __init__(self, entities):
        self.entities = entities

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        entity = self.entities[index]
        entity_token_ids = torch.tensor(entity["text_ids"]).long()
        entity_masks = torch.tensor(entity["text_masks"]).long()
        return entity_token_ids, entity_masks


# For embedding all the samples during inference
class SampleSet(Dataset):
    def __init__(self, samples, max_len, tokenizer, use_title=False):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.use_title = use_title
        # [2] is token id of '[unused1]' for bert tokenizer
        self.TT = [2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mention = self.samples[index]
        if self.use_title:
            title = mention["title"]
            title_ids = self.TT + title
        else:
            title_ids = []
        # CLS + mention ids + TT + title ids
        mention_title_ids = mention["text"] + title_ids
        mention_ids = (
            mention_title_ids
            + [self.tokenizer.pad_token_id] * (self.max_len - len(mention_title_ids))
        )[: self.max_len]
        mention_masks = (
            [1] * len(mention_title_ids) + [0] * (self.max_len - len(mention_title_ids))
        )[: self.max_len]
        mention_token_ids = torch.tensor(mention_ids).long()
        mention_masks = torch.tensor(mention_masks).long()
        return mention_token_ids, mention_masks


def make_single_loader(data_set, bsz, shuffle):
    loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader


def get_loaders(
    samples, entities, max_len, tokenizer, mention_bsz, entity_bsz, use_title
):
    #  get all samples and entity dataloaders
    samples_set = SampleSet(samples, max_len, tokenizer, use_title)
    entity_set = EntitySet(entities)
    entity_loader = make_single_loader(entity_set, entity_bsz, False)
    samples_loader = make_single_loader(samples_set, mention_bsz, False)

    return samples_loader, entity_loader


def get_embeddings(loader, model, is_sample, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks = batch
            k1, k2 = (
                ("mention_token_ids", "mention_masks")
                if is_sample
                else ("entity_token_ids", "entity_masks")
            )
            kwargs = {k1: input_ids, k2: input_masks}
            j = 0 if is_sample else 2
            embed = model(**kwargs)[j].detach()
            embeddings.append(embed.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    model.train()
    return embeddings


def get_hard_negative(
    mention_embeddings, all_entity_embeds, k, max_num_postives, use_gpu_index=False
):
    index = faiss.IndexFlatIP(all_entity_embeds.shape[1])
    if use_gpu_index:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(all_entity_embeds)
    scores, hard_indices = index.search(mention_embeddings, k + max_num_postives)
    del mention_embeddings
    del index
    return hard_indices, scores


def save_candidates(tokenizer, samples, topk_candidates, entity_map, out_dir):
    # save results for reader training
    assert len(samples) == len(topk_candidates)
    out_path = os.path.join(out_dir, "result.json")
    entity_titles = np.array(list(entity_map.keys()))
    fout = open(out_path, "w")
    for i in range(len(samples)):
        sample = samples[i]
        m_candidates = topk_candidates[i].tolist()
        candidate_titles = entity_titles[m_candidates]
        item = {
            "doc_id": sample["doc_id"],
            "mention_idx": i,
            "candidates": m_candidates,
            "title_ids": sample["title"],
            "token_ids": sample["text"],
            "title_text": tokenizer.decode(sample["title"]),
            "token_text": tokenizer.decode(sample["text"]),
            "offset": sample["offset"],
            "candidate_titles": candidate_titles,
        }
        print(item)
        fout.write("%s\n" % json.dumps(item))
    fout.close()


def main(args):
    start_time = datetime.now()
    set_seeds(args)
    # configure logger
    best_val_perf = float("-inf")
    logger = Logger(args.model + ".log", on=True)
    logger.log(str(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.log(f"Using device: {str(device)}", force=True)
    samples = load_data(args.data_dir)
    max_num_positives = args.k - args.num_cands
    config = {
        "top_k": 100,
        "biencoder_model": args.pretrained_path + "biencoder_wiki_large.bin",
        "biencoder_config": args.pretrained_path + "biencoder_wiki_large.json",
    }

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    tokenized_samples = tokenize_original_text(
        samples, tokenizer, {"instance_length": 32, "stride": 16}
    )
    entities = load_entities(args.kb_dir)
    entity_map = get_entity_map(entities)

    model = load_model(
        config["biencoder_config"],
        args.model,
        device,
        args.type_loss,
        args.blink,
    )
    model.to(device)
    model.eval()
    all_cands_embeds = np.load(args.cands_embeds_path)
    logger.log("getting test mention embeddings ...")
    samples_loader, entity_loader = get_loaders(
        tokenized_samples,
        entities,
        args.max_len,
        tokenizer,
        args.mention_bsz,
        args.entity_bsz,
        args.use_title,
    )
    test_mention_embeds = get_embeddings(samples_loader, model, True, device)
    topk_candidates, scores_k = get_hard_negative(
        test_mention_embeds, all_cands_embeds, args.k, 0, args.use_gpu_index
    )
    save_candidates(
        tokenizer, tokenized_samples, topk_candidates, entity_map, args.out_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="the directory of the wikipedia pretrained models",
    )
    parser.add_argument(
        "--blink", action="store_true", help="use BLINK pretrained model?"
    )
    parser.add_argument(
        "--max_len", type=int, default=100, help="max length of the mention input "
    )
    parser.add_argument("--k", type=int, default=100, help="recall@k when evaluate")
    parser.add_argument("--use_title", action="store_true", help="use title?")
    parser.add_argument("--data_dir", type=str, help="the data directory")
    parser.add_argument("--kb_dir", type=str, help="the knowledge base directory")
    parser.add_argument("--out_dir", type=str, help="the output saving directory")
    parser.add_argument("--B", type=int, default=16, help="the batch size per gpu")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed [%(default)d]"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num workers [%(default)d]"
    )
    parser.add_argument(
        "--clip", type=float, default=1, help="gradient clipping [%(default)g]"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="num logging steps [%(default)d]",
    )
    parser.add_argument(
        "--gpus", default="", type=str, help="GPUs separated by comma [%(default)s]"
    )
    parser.add_argument(
        "--rands_ratio",
        default=1.0,
        type=float,
        help="the ratio of random candidates and hard",
    )
    parser.add_argument(
        "--num_cands", default=64, type=int, help="the total number of candidates"
    )
    parser.add_argument("--mention_bsz", type=int, default=512, help="the batch size")
    parser.add_argument("--entity_bsz", type=int, default=512, help="the batch size")
    parser.add_argument("--use_gpu_index", action="store_true", help="use gpu index?")
    parser.add_argument(
        "--type_loss",
        type=str,
        choices=["log_sum", "sum_log", "sum_log_nce", "max_min"],
        help="type of multi-label loss ?",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) "
        "instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', "
        "'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--cands_embeds_path", type=str, help="the directory of candidates embeddings"
    )
    parser.add_argument(
        "--use_cached_embeds",
        action="store_true",
        help="use cached candidates embeddings ?",
    )
    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # Sets torch.cuda behavior
    main(args)

    # python ./simple.py --model ./models/retriever.pt --type_loss sum_log_nce --data_dir ./input/ --kb_dir ./models/data/kb/ --k 100 --num_cands 64  --pretrained_path ./models/  --max_len 42  --mention_bsz 512 --entity_bsz 512  --B 4  --rands_ratio 0.9 --logging_step 100 --out_dir ./models/retriever_output --cands_embeds_path ./models/candidate_embeds.npy --blink  --use_title --gpus 0
