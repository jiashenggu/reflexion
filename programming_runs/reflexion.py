from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, 1)

            dataset_name = item["task_id"].split(
                "/")[0].replace("Eval", "").lower()
            # first attempt

            retrieved_apis, code_prefix = item["prompt"].split("# [end]")
            retrieved_apis = retrieved_apis.split("# [start]")[1].strip()
            retrieved_functions = ""
            apis = retrieved_apis.split("\n")
            code_prefix = code_prefix.strip()
            cur_func_impl = gen.func_impl(
                retrieved_apis, retrieved_functions, code_prefix, model, "simple")
            # import re

            # # Define the pattern to extract content between # [start] and # [end]
            # pattern = r'# \[start\](.*?)# \[end\]'

            # # Extracting the content
            # matched_content = re.search(pattern, item["prompt"], re.DOTALL)

            # if matched_content:
            #     apis = matched_content.group(1).strip().split("\n")
            # else:
            #     apis = []

            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            # if solved, exit early
            if is_passing:
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing
                num_success += int(is_passing)
                break

            # use self-reflection to iteratively improve
            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                # get self-reflection
                reflection = gen.self_reflection(
                    cur_func_impl, cur_feedback, model)
                reflections += [reflection]
                # pattern = r"# \[start\].*?# \[end\]"
                # update_prompt = item["prompt"].replace(pattern, replacement, 1)

                retrieved_apis, code_prefix = item["prompt"].split("# [end]")
                retrieved_apis = retrieved_apis.split("# [start]")[1].strip()

                from retrieval import get_topk_apis
                json_file_paths = [
                    f"/home/gujiasheng/reflexion/api_coder/{dataset_name}_api.json"]
                # json_file_paths = [f"/home/gujiasheng/reflexion/api_coder/{dataset_name}_api.json",
                #                    f"/home/gujiasheng/reflexion/api_coder/{dataset_name}_github.json"]
                retrieved_apis = get_topk_apis(
                    json_file_paths, cur_func_impl+cur_feedback, top_k=5)

                # from run_live import retrieve_repo
                # args = {
                #     "query": cur_func_impl+cur_feedback,
                #     "prompt_style": "style-3",
                #     "issue_url": ["https://github.com/pytorch/data/issues/1169"],
                #     "base_commit": [None],
                #     "max_context_length": 2000,
                #     "document_encoding_func": "file_name_and_contents",
                #     "root_dir": "./run_live_data",
                #     "include_readmes": False
                # }

                # instance = retrieve_repo(**args)
                retrieved_functions = ""
                # for v in instance['file_contents'].values():
                #     retrieved_functions += v + "\n\n\n"

                apis = retrieved_apis.split("\n")
                code_prefix = code_prefix.strip()
                cur_func_impl = gen.func_impl(
                    retrieved_apis=retrieved_apis,
                    retrieved_functions=retrieved_functions,
                    func_sig=code_prefix,
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )

                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # check if all internal unit tests pass
                is_passing, cur_feedback, _ = exe.execute(
                    cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                # if solved, check if it passes the real tests, exit early
                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                    break

                cur_iter += 1
            cur_pass += 1

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["internal_test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item["apis"] = apis
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
