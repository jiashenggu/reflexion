from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List
from retrieval import Retriever


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    # retriever = Retriever("bm25", topk=5)
    retriever = Retriever(
        "embedding", "/ML-A100/home/gujiasheng/bge-small-en", topk=5, topk_bm25=100
    )
    dataset_name = dataset[0]["task_id"].split("/")[0].replace("Eval", "").lower()
    json_file_paths = [
        f"/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_api_1.json"
    ]
    # json_file_paths = [f"/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_api.json",
    #                    f"/ML-A100/home/gujiasheng/reflexion/api_coder/{dataset_name}_github.json"]
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        apis = []
        cur_func_impl = ""
        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item["visible_tests"]
            else:
                # tests_i = gen.internal_tests(item["prompt"], model, 1)
                tests_i = []

            # first attempt

            retrieved_apis, code_prefix = item["prompt"].split("# [end]")
            retrieved_apis = retrieved_apis.split("# [start]")[1].strip()
            reflection_query = ""
            retrieved_apis = retriever.get_topk_apis(json_file_paths, code_prefix)
            code_prefix = code_prefix.strip()
            item["prompt"] = code_prefix
            retrieved_source_code = ""
            apis.append(retrieved_apis.split("\n\n\n"))

            cur_func_impl = gen.func_impl(
                retrieved_apis, retrieved_source_code, code_prefix, model, "simple"
            )

            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            # if solved, exit early
            if is_passing:
                is_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10
                )
                num_success += int(is_solved)
                break

            # use self-reflection to iteratively improve
            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                # get self-reflection
                reflection = gen.self_reflection(cur_func_impl, cur_feedback, model)
                reflections += [reflection]
                reflection_query = cur_func_impl + cur_feedback
                # reflection_query = gen.self_reflection_retrieval(cur_func_impl, cur_feedback, model)
                retrieved_apis = retriever.get_topk_apis(
                    json_file_paths, reflection_query
                )

                retrieved_source_code = ""

                apis.append(retrieved_apis.split("\n\n\n"))

                cur_func_impl = gen.func_impl(
                    retrieved_apis=retrieved_apis,
                    retrieved_source_code=retrieved_source_code,
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
                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                # if solved, check if it passes the real tests, exit early
                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10
                    )
                    if is_passing:
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                    break

                cur_iter += 1
            cur_pass += 1

        item["is_solved"] = is_solved
        if is_solved:
            print("----------------------------------------------")
            print("Solved!")
            print("----------------------------------------------")
        else:
            print("----------------------------------------------")
            print("Not solved!")
            print("----------------------------------------------")
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["internal_test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item["apis"] = apis
        item["reflection_query"] = reflection_query
        write_jsonl(log_path, [item], append=True)

        print_v(f"completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}")
