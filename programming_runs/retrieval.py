import re
import subprocess
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import random
import json
from FlagEmbedding import FlagModel
import os
import tiktoken
from transformers import LlamaTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"


def cl100k(text, tokenizer):
    return tokenizer.encode(text, disallowed_special=())


def llama(text, tokenizer):
    return tokenizer(text, add_special_tokens=False, return_attention_mask=False)[
        "input_ids"
    ]


TOKENIZER_FUNCS = {
    "cl100k": (tiktoken.get_encoding("cl100k_base"), cl100k),
    "llama": (LlamaTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K"), llama),
}
tokenizer, tokenizer_func = TOKENIZER_FUNCS["cl100k"]


class Retriever:
    def __init__(self, retrieve_type, model_path=None, topk=5, topk_bm25=None):
        self.retrieve_type = retrieve_type
        if self.retrieve_type == "embedding":
            self.model = FlagModel(
                model_path,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                use_fp16=True,
            )
        self.topk = topk
        self.topk_bm25 = topk_bm25

    def extract_code_between_triple_quotes(self, code):
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, code, re.DOTALL)
        return matches

    def extract_python_code(self, text):
        pattern = r"(?<!\w)def\s+\w+\(.*?\):\s*[\n\s\S]+?(?=^\w|\Z)"
        matches = re.findall(pattern, text, re.MULTILINE)
        return matches

    def get_topk_documents_from_json(
        self, json_file_paths, query, max_context_length=3000
    ):
        # Read JSON file
        data = []

        for json_file_path in json_file_paths:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                for line in json_file:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except json.decoder.JSONDecodeError:
                        print(f"Skipping invalid JSON: {line.strip()}")

        # Extract text and IDs from JSON data
        text_ids = [entry["text_id"] for entry in data]
        texts = [entry["text"].lower().split() for entry in data]
        bm25_model = BM25Okapi(texts)

        query_tokens = query.lower().split()
        bm25_scores = bm25_model.get_scores(query_tokens)
        text_scores = list(zip(text_ids, bm25_scores))
        sorted_texts = sorted(text_scores, key=lambda x: x[1], reverse=True)

        topk_texts = []
        token_count = 0
        k_limit = self.topk  # Set the maximum value of k to 3
        for text_id, bm25_score in sorted_texts[: self.topk]:
            # Retrieve the corresponding text content
            text_content = next(
                entry["text"] for entry in data if entry["text_id"] == text_id
            )

            if (
                token_count + len(tokenizer_func(text_content, tokenizer))
                <= max_context_length
                and len(topk_texts) < k_limit
            ):
                token_count += len(tokenizer_func(text_content, tokenizer))
                topk_texts.append((text_content, text_id, bm25_score))
            else:
                continue

        return topk_texts

    def get_topk_documents_from_json_embedding(
        self, json_file_paths, query, max_context_length=3000
    ):
        # Read JSON file
        data = []
        if self.topk_bm25 is None:
            for json_file_path in json_file_paths:
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    for line in json_file:
                        try:
                            entry = json.loads(line)
                            data.append(entry)
                        except json.decoder.JSONDecodeError:
                            print(f"Skipping invalid JSON: {line.strip()}")
        else:
            topk_bm25_texts = self.get_topk_documents_from_json(
                json_file_paths, query, max_context_length
            )
            data = []

            for text_content, text_id, _ in topk_bm25_texts:
                entry = {}
                entry["text_id"] = text_id
                entry["text"] = text_content
                data.append(entry)

        text_ids = [entry["text_id"] for entry in data]
        texts = [entry["text"] for entry in data]
        q_embeddings = self.model.encode_queries([query])
        p_embeddings = self.model.encode(texts)
        scores = q_embeddings @ p_embeddings.T

        text_scores = list(zip(text_ids, scores[0]))
        sorted_texts = sorted(text_scores, key=lambda x: x[1], reverse=True)

        topk_texts = []
        token_count = 0
        k_limit = self.topk  # Set the maximum value of k to 5
        for text_id, bm25_score in sorted_texts[: self.topk]:
            # Retrieve the corresponding text content
            text_content = next(
                entry["text"] for entry in data if entry["text_id"] == text_id
            )
            if (
                token_count + len(tokenizer_func(text_content, tokenizer))
                <= max_context_length
                and len(topk_texts) < k_limit
            ):
                token_count += len(tokenizer_func(text_content, tokenizer))
                topk_texts.append((text_content, text_id, bm25_score))
            else:
                continue

        return topk_texts

    # def get_topk_documents(self, dataset_name):
    #     data = []

    #     with open(f'/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_comment.json', "r") as fp:
    #         for line in fp.readlines():
    #             data.append(json.loads(line))
    #     print("Processing the data.")
    #     for line in tqdm(data):
    #         query = line["text"]
    #         print(query)

    #         # Example usage:
    #         json_file_paths = [f"/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_api.json",
    #                            f"/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_github.json"]
    #         if self.retrieve_type == "bm25":
    #             topk_documents = self.get_topk_documents_from_json(
    #                 json_file_paths, query)
    #         elif self.retrieve_type == "embedding":
    #             topk_documents = self.get_topk_documents_from_json_embedding(
    #                 json_file_paths, query)

    #         # Print the top-k documents
    #         document_str = ""
    #         for text_content, text_id, bm25_score in topk_documents:
    #             document_str += f"{text_content}\n"
    #         apis = "# [start]\n" + document_str + "# [end]"
    #         print(apis)
    #         return apis

    def get_topk_apis(self, json_file_paths, query):
        if self.retrieve_type == "bm25":
            topk_documents = self.get_topk_documents_from_json(json_file_paths, query)
        elif self.retrieve_type == "embedding":
            topk_documents = self.get_topk_documents_from_json_embedding(
                json_file_paths, query
            )
        document_str = ""
        for text_content, text_id, score in topk_documents:
            document_str += f"{text_content}\n\n\n"
        document_str = document_str.rstrip("\n")
        return document_str


if __name__ == "__main__":
    dataset_name = "torchdata"
    retriever = Retriever("embedding", "/ML-A100/home/gujiasheng/bge-small-en")
    retriever.get_topk_documents(dataset_name)
    json_file_paths = [
        f"/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_api.json",
        f"/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_github.json",
    ]
    retriever.get_topk_apis(
        json_file_paths, "How to augment the datapipe by repeating it six times."
    )
