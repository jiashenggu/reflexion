"""Tools to generate from OpenAI prompts."""

from FlagEmbedding import FlagModel
import asyncio
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
from typing import Any

import json
import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from tqdm import tqdm
import json
from rank_bm25 import BM25Okapi

import subprocess
import json
import re


def perturbation_prompt(question, instruction):
    message = [
        {"role": "system", "content": instruction},

        {"role": "user", "content": question},
    ]
    return message


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list(dict()),
    temperature: float,
    # max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict():
    async with limiter:
        for _ in range(10):
            try:
                return await openai.ChatCompletion.acreate(
                    engine=model,
                    messages=messages,
                    temperature=temperature,
                    # max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 20 seconds."
                )
                sleep_time = random.randint(10, 20)
                await asyncio.sleep(sleep_time)
            except asyncio.exceptions.TimeoutError or openai.error.Timeout or asyncio.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
            except:
                logging.warning(
                    "Unknown OpenAI API error. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    api_key: str,
    messages,
    engine_name: str,
    temperature: float,
    # max_tokens: int,
    top_p: float,
    requests_per_minute: int = 300,
) -> list():
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = api_key
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        await _throttled_openai_chat_completion_acreate(
            model=engine_name,
            messages=message,
            temperature=temperature,
            # max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return [x["choices"][0]["message"]["content"] for x in responses]


def get_topk_documents_from_json(json_file_paths, query, max_char_limit=5000):
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
    char_count = 0
    k_limit = 5  # Set the maximum value of k to 3
    for text_id, bm25_score in sorted_texts:
        # Retrieve the corresponding text content
        text_content = next(entry["text"]
                            for entry in data if entry["text_id"] == text_id)
        char_count += len(text_content)
        if char_count <= max_char_limit and len(topk_texts) < k_limit:
            topk_texts.append((text_content, text_id, bm25_score))
        else:
            break

    return topk_texts


model = FlagModel('/ML-A100/home/gujiasheng/bge-small-en',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation


def get_topk_documents_from_json_bpe(json_file_paths, query, top_k=3, max_char_limit=5000):
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
    texts = [entry["text"] for entry in data]

    q_embeddings = model.encode_queries([query])
    p_embeddings = model.encode(texts)
    scores = q_embeddings @ p_embeddings.T

    text_scores = list(zip(text_ids, scores[0]))

    sorted_texts = sorted(text_scores, key=lambda x: x[1], reverse=True)

    topk_texts = []
    char_count = 0
    k_limit = top_k  # Set the maximum value of k to 5
    for text_id, bm25_score in sorted_texts:
        # Retrieve the corresponding text content
        text_content = next(entry["text"]
                            for entry in data if entry["text_id"] == text_id)
        char_count += len(text_content)
        if char_count <= max_char_limit and len(topk_texts) < k_limit:
            topk_texts.append((text_content, text_id, bm25_score))
        else:
            break

    return topk_texts


def extract_code_between_triple_quotes(code):
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, code, re.DOTALL)
    return matches


def extract_python_code(text):
    pattern = r"(?<!\w)def\s+\w+\(.*?\):\s*[\n\s\S]+?(?=^\w|\Z)"
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches


def get_topk_documents(dataset_name):
    data = []
    with open(f'/home/gujiasheng/reflexion/api_coder/{dataset_name}_comment.json', "r") as fp:
        for line in fp.readlines():
            data.append(json.loads(line))
    print("Processing the data.")
    for line in tqdm(data):
        query = line["text"]
        print(query)

        # Example usage:

        json_file_paths = [f"/home/gujiasheng/reflexion/api_coder/{dataset_name}_api.json",
                           f"/home/gujiasheng/reflexion/api_coder/{dataset_name}_github.json"]
        topk_documents = get_topk_documents_from_json(json_file_paths, query)

        # Print the top-k documents
        document_str = ""
        for text_content, text_id, bm25_score in topk_documents:
            document_str += f"{text_content}\n"
        apis = "# [start]\n"+document_str+"# [end]"
        print(apis)
        return apis


def get_topk_apis(json_file_paths, query, top_k=3):
    topk_documents = get_topk_documents_from_json_bpe(
        json_file_paths, query, top_k=top_k, max_char_limit=5000)
    document_str = ""
    for text_content, text_id, bm25_score in topk_documents:
        document_str += f"{text_content}\n"
    return document_str


if __name__ == "__main__":
    dataset_name = "torchdata"
    # get_topk_documents(dataset_name)
    json_file_paths = [f"/home/gujiasheng/reflexion/api_coder/{dataset_name}_api.json",
                       f"/home/gujiasheng/reflexion/api_coder/{dataset_name}_github.json"]
    get_topk_apis(
        json_file_paths, "How to augument the datapipe by repeating it six times.")
