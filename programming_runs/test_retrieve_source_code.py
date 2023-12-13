from retrieve_source_code import retrieve_repo
args = {
    "query": "good",
    "prompt_style": "style-3",
    "issue_url": ["https://github.com/pytorch/data/issues/1169"],
    "base_commit": [None],
    "max_context_length": 4096,
    "document_encoding_func": "file_name_and_contents",
    "root_dir": "./run_live_data",
    "include_readmes": False
}

instance = retrieve_repo(**args)
retrieved_source_code = ""
for v in instance['file_contents'].values():
    retrieved_source_code += v + "\n\n\n"
print(retrieved_source_code)
