from __future__ import annotations

import time

from hackathon.evaluation.evaluate import get_runnable_problems, run_agents_and_catch_logs, run_swebench_evaluation
from getpass import getuser
import ast
import tiktoken
import numpy as np
encoder = tiktoken.encoding_for_model("gpt-4")

def get_ft_message_token_counts(data):
    token_counts = []
    for list in data:
        token_counts.append(
            int(len(encoder.encode(list["messages"][0]["content"]+list["messages"][1]["content"])))
        )
    return {
        "min":int(np.min(token_counts)),
        "max":int(np.max(token_counts)),
        "mean":int(np.mean(token_counts)),
        "median":int(np.median(token_counts)),
        "Q1":int(np.percentile(token_counts, 25)),
        "Q3":int(np.percentile(token_counts, 75)),
        "P90":int(np.percentile(token_counts, 90)),
    }


def fix_jsonl(file_path):
    fixed_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                python_obj = ast.literal_eval(line.strip())
                json_obj = json.loads(json.dumps(python_obj))
                fixed_data.append(json_obj)
            except (ValueError, SyntaxError):
                pass
    
    return fixed_data

if __name__ == "__main__":
    from datasets import load_dataset

    d = load_dataset("princeton-nlp/SWE-bench_Lite")#_Lite
    # mode = ["mini", "sonnet", "L3.1-70b-Together", "L3.1-70b-Baseten", "L3.1-405b-Baseten", "L3.1-70b-Groq", "open-pipe-llama8b"][2]
    # if mode == "mini":
    #     model_name = "gpt-4o-mini"
    #     cost_limit = 0.21
    # elif mode == "sonnet":
    #     model_name = "claude-3-5-sonnet-20240620"
    #     cost_limit = 1.5
    # elif mode == "L3.1-70b-Together":
    #     model_name = "L3.1-70b-Together"
    #     cost_limit = 0.25
    # elif mode == "L3.1-70b-Baseten":
    #     model_name = "L3.1-70b-BaseTen"
    #     cost_limit = 1.0
    # elif mode == "L3.1-405b-Baseten":
    #     model_name = "L3.1-405b-BaseTen"
    #     cost_limit = 1.0
    # elif mode == "L3.1-70b-Groq":
    #     model_name = "L3.1-70b-Groq"
    #     cost_limit = 1.0
    
    model_name = "open-pipe-llama70b"
    cost_limit = 1.0

    run_agent = True
    evaluate_agent = False
    
    split = "dev"
    question_ids = ["django__django-14855"]#["astropy__astropy-14995","django__django-14382"]
    question_ids = [
        "astropy__astropy-14995",
        "django__django-11039",
        "django__django-11099",
        "django__django-11133",
        "django__django-12453",
        "django__django-12983",
        "django__django-13658",
        "django__django-14382",
        "django__django-14855",
    ]
    first_question_index = 0
    last_question_index = 23
    question_ids = [
        d[split][question_index]["instance_id"]
        for question_index in range(first_question_index, last_question_index)
    ]

    runnable_problems_by_split = get_runnable_problems(
        f"trajectories/{getuser()}/{model_name}__SWE-bench_Lite__default__t-0.00__p-0.95__c-{cost_limit:.2f}__install-1",dataset_name="princeton-nlp/SWE-bench_Lite"
    )
    print("Model name: ", model_name)
    print("Split: ", split)
    print({k: len(v) for k, v in runnable_problems_by_split.items()})
    t0_agent = time.time()
    if run_agent:
        
        run_agents_and_catch_logs(
            model_name=model_name, instance_ids=question_ids, instance_cost_limit=cost_limit, split=split
        )
    print("Time taken to run agent: ", time.time() - t0_agent)
    if evaluate_agent:
        import time

        t0 = time.time()
        splits = ["dev", "test"]
        for split in splits:
            print("Running evaluation for split: ", split)
            successful_ids, failed_ids, information_by_instance = run_swebench_evaluation(
                predictions_path_override=None,
                model_name=model_name,
                full_dataset_name="princeton-nlp/SWE-bench_Lite",
                cost_limit=cost_limit,
                temperature=0.00,
                top_p=0.95,
                run_id="test",
                split=split,
                max_workers=2,
                full_dataset=d,
                test_ids=question_ids,
            )
            import os
            import glob
            import shutil
            import json

            finetuning_dir = "hackathon/finetuning"
            jsonl_files = glob.glob(os.path.join(finetuning_dir, "*.jsonl"))
            print(information_by_instance.keys())
            for id in successful_ids:
                for jsonl_file in jsonl_files:
                    if id in jsonl_file:
                        data = fix_jsonl(jsonl_file)
                        full_datum = {
                            "instance_id": id,
                            "ft_messages": data,
                            "ft_message_token_counts": get_ft_message_token_counts(data),
                            "information": information_by_instance[id] if id in information_by_instance else {},
                        }
                        with open(f"hackathon/data/successful/{id}_{model_name}.json", 'w') as f:
                            json.dump(full_datum, f)
            for jsonl_file in jsonl_files:
                if os.path.exists(jsonl_file):
                    data = fix_jsonl(jsonl_file)
                    id = jsonl_file.split("_training_data")[0].split("_install-1_")[1]
                    full_datum = {
                        "instance_id": id,
                        "ft_messages": data,
                        "ft_message_token_counts": get_ft_message_token_counts(data),
                        "information": information_by_instance[id] if id in information_by_instance else {},
                    }
                    with open(f"hackathon/data/failed/{id}_{model_name}.json", 'w') as f:
                        json.dump(full_datum, f)

        print("Time taken to evaluate runs: ", time.time() - t0)
        #Captured 5551 lines of logs???
