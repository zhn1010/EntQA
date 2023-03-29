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
from collections import Counter, defaultdict


def load_data(data_dir):
    result = []
    with open(os.path.join(data_dir, "samples.jsonl")) as f:
        for line in f:
            result.append(json.loads(line))
    return result


def tokenize_original_text(raw_data, tokenizer, args):
    data = []
    tokenized_raw_data = {}
    for d in raw_data:
        orig_text = d["text"]
        orig_title = d["title"]
        text = tokenizer.tokenize(orig_text)
        doc_id = d["doc_id"]
        tokenized_raw_data[doc_id] = {"orig_text": orig_text, "tokenized_text": text}
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
    return data, tokenized_raw_data


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


def load_retriever_model(config_path, model_path, device, type_loss, blink=True):
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


def get_retriever_loader(
    samples, entities, max_len, tokenizer, mention_bsz, entity_bsz, use_title
):
    #  get all samples and entity dataloaders
    samples_set = SampleSet(samples, max_len, tokenizer, use_title)
    entity_set = EntitySet(entities)
    samples_loader = make_single_loader(samples_set, mention_bsz, False)

    return samples_loader  # , entity_loader


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
    return hard_indices  # , scores


def prepare_candidates(tokenizer, samples, topk_candidates, entity_map):
    # save results for reader training
    assert len(samples) == len(topk_candidates)
    entity_titles = np.array(list(entity_map.keys()))
    candidates = []
    for i in range(len(samples)):
        sample = samples[i]
        m_candidates = topk_candidates[i].tolist()
        candidate_titles = entity_titles[m_candidates]
        item = {
            "doc_id": sample["doc_id"],
            "mention_idx": i,
            "offset": sample["offset"],
            "candidates": m_candidates,
            "title_ids": sample["title"],
            "token_ids": sample["text"],
            "title_text": tokenizer.decode(sample["title"]),
            "token_text": tokenizer.decode(sample["text"]),
            "candidate_titles": candidate_titles.tolist(),
        }
        candidates.append(item)
    return candidates


# --------------------------------- READER -------------------------------------#


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


def load_reader_model(
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
        )


def make_single_loader(data_set, bsz, shuffle):
    loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader


def get_reader_loaders(
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
    model,
    device,
    loader,
    k,
    samples,
    filter_span=True,
    no_multi_ents=False,
    do_rerank=True,
):
    model.eval()
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
        ps = torch.cat(ps, 0)
    raw_predicts = get_predicts(ps, k, filter_span, no_multi_ents)
    assert len(raw_predicts) == len(samples)
    return raw_predicts


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
def get_sample_results(predicts, samples):
    # save_path = os.path.join(results_dir, "reader_results.jsonl")
    results = []
    for predict, sample in zip(predicts, samples):
        result = {}
        result["doc_id"] = sample["doc_id"]
        result["offset"] = sample["offset"]
        result["text"] = sample["token_text"]
        result["predicts"] = predict
        results.append(result)
    return results
    # with open(save_path, "w") as f:
    #     for r in results:
    #         f.write("%s\n" % json.dumps(r))


def get_sample_docs(sample_results, tokenized_raw_data, entity_map):
    results = {}
    tuple_list = []
    for sample in sample_results:
        offset = sample["offset"]
        p = sample["predicts"]
        if len(p) == 0:
            continue
        for r in p:
            result = (sample["doc_id"], r[0] + offset, r[1] + offset, r[2])
            tuple_list.append(result)
    counted = Counter(tuple_list)
    grouped_tuple_list = defaultdict(list)
    for element, count in counted.items():
        if count > 1:
            doc_id, begin, end, entity_title = element
            entity_text = " ".join(
                tokenized_raw_data[doc_id]["tokenized_text"][begin - 1 : end]
            ).replace("##", "")
            grouped_tuple_list[doc_id].append(
                (begin, end, entity_title, count, entity_map[entity_title], entity_text)
            )
    assert len(tokenized_raw_data) == len(grouped_tuple_list)
    return grouped_tuple_list


def save_doc_results(doc_results, tokenized_raw_data, out_dir):
    save_path = os.path.join(out_dir, "reader_results.jsonl")
    results = []
    for doc_id, predictions in doc_results.items():
        result = {}
        result["doc_id"] = doc_id
        result["text"] = tokenized_raw_data[doc_id]["orig_text"]
        result["tokenized_text"] = tokenized_raw_data[doc_id]["tokenized_text"]
        result["predicts"] = [
            {
                "start_token": item[0],
                "end_token": item[1],
                "entity_title": item[2],
                "entity_id": item[4],
                "text": item[5],
                "count": item[3],
            }
            for item in predictions
        ]
        results.append(result)

    with open(save_path, "w") as f:
        for r in results:
            f.write("%s\n" % json.dumps(r))


# ---------------------------------------------------------------------------------------------------#


def main(args):
    set_seeds(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    raw_data = load_data(args.data_dir)
    biencoder_config = args.pretrained_path + "biencoder_wiki_large.json"

    retriever_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    tokenized_samples, tokenized_raw_data = tokenize_original_text(
        raw_data, retriever_tokenizer, {"instance_length": 32, "stride": 16}
    )
    entities = load_entities(args.kb_dir)
    entity_map = get_entity_map(entities)

    # -------------------------------------- RETRIEVER ------------------------------------------- #

    retriever_model = load_retriever_model(
        biencoder_config,
        args.retriver_model,
        device,
        args.type_loss,
        args.blink,
    )
    retriever_model.to(device)
    retriever_model.eval()
    all_cands_embeds = np.load(args.cands_embeds_path)
    samples_loader = get_retriever_loader(
        tokenized_samples,
        entities,
        args.max_len,
        retriever_tokenizer,
        args.mention_bsz,
        args.entity_bsz,
        args.use_title,
    )
    test_mention_embeds = get_embeddings(samples_loader, retriever_model, True, device)
    topk_candidates = get_hard_negative(
        test_mention_embeds,
        all_cands_embeds,
        args.retriever_recall_at_k,
        0,
        args.use_gpu_index,
    )
    candidates = prepare_candidates(
        retriever_tokenizer, tokenized_samples, topk_candidates, entity_map
    )

    del test_mention_embeds
    del retriever_model
    del all_cands_embeds
    torch.cuda.empty_cache()

    # -------------------------------------- READER ------------------------------------------- #

    reader_model, reader_tokenizer = load_reader_model(
        args.reader_model,
        args.type_encoder,
        device,
        args.type_span_loss,
        args.do_rerank,
        args.type_rank_loss,
        args.max_answer_len,
        args.max_passage_len,
    )
    loader = get_reader_loaders(
        reader_tokenizer,
        candidates,
        entities,
        args.L,
        args.C,
        args.B,
        args.use_title,
    )
    reader_model.to(device)
    args.n_gpu = torch.cuda.device_count()
    dp = args.n_gpu > 1
    if dp:
        reader_model = nn.DataParallel(reader_model)
    reader_model.eval()
    raw_predicts = get_raw_results(
        reader_model,
        device,
        loader,
        args.k,
        candidates,
        args.filter_span,
        args.no_multi_ents,
        args.do_rerank,
    )
    pruned_preds = prune_predicts(raw_predicts, args.thresd)
    predicts = transform_predicts(pruned_preds, entities, candidates)
    sample_results = get_sample_results(predicts, candidates)  # , args.out_dir)
    doc_results = get_sample_docs(sample_results, tokenized_raw_data, entity_map)
    save_doc_results(doc_results, tokenized_raw_data, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------------- RETRIEVER ------------------------------------ #

    parser.add_argument("--retriver_model", type=str, help="retriver model path")
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
    parser.add_argument(
        "--retriever_recall_at_k", type=int, default=100, help="recall@k when evaluate"
    )
    parser.add_argument("--use_title", action="store_true", help="use title?")
    parser.add_argument("--data_dir", type=str, help="the data directory")
    parser.add_argument("--kb_dir", type=str, help="the knowledge base directory")
    parser.add_argument("--out_dir", type=str, help="the output saving directory")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed [%(default)d]"
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
        "--cands_embeds_path", type=str, help="the directory of candidates embeddings"
    )
    parser.add_argument(
        "--use_cached_embeds",
        action="store_true",
        help="use cached candidates embeddings ?",
    )

    # ----------------------------------- READER ----------------------------------- #

    parser.add_argument("--reader_model", type=str, help="reader model path")
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

# python ./simple.py --retriver_model ./models/retriever.pt  --pretrained_path ./models/ --blink --max_len 42 --retriever_recall_at_k 100 --use_title  --data_dir ./input/ --kb_dir ./models/data/kb/ --out_dir ./models/reader_retriever_output --gpus 0 --rands_ratio 0.9 --num_cands 64 --mention_bsz 512 --entity_bsz 512 --type_loss sum_log_nce --cands_embeds_path ./models/candidate_embeds.npy --reader_model ./models/reader.pt --C 100  --B 5  --L 180 --thresd  0.05  --k 3  --max_passage_len 32  --filter_span  --type_encoder squad2_electra_large  --type_span_loss sum_log  --type_rank_loss sum_log  --do_rerank
