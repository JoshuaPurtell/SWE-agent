
from hackathon.evaluation.evaluate import get_runnable_problems, run_agents_and_catch_logs, run_swebench_evaluation
from getpass import getuser
import ast
import tiktoken
import numpy as np
import os
import glob
import shutil
import json
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

def main(dataset_name="princeton-nlp/SWE-bench_Lite", model_name="gpt-4o-mini", cost_limit=0.21, split="dev", first_question_index=80, last_question_index=85):
    from datasets import load_dataset
    import time
    print(dataset_name)
    d = load_dataset(dataset_name)
    run_agent = True
    evaluate_agent = True
    question_ids = [
        d[split][question_index]["instance_id"]
        for question_index in range(first_question_index, last_question_index)
    ]
    runnable_problems_by_split = get_runnable_problems(
        f"trajectories/{getuser()}/{model_name}__{dataset_name}__default__t-0.00__p-0.95__c-{cost_limit:.2f}__install-1"
    )
    print("Model name: ", model_name)
    print("Split: ", split)
    print({k: len(v) for k, v in runnable_problems_by_split.items()})
    t0_agent = time.time()
    if run_agent:
        chunk_size = 10
        for i in range(0, len(question_ids), chunk_size):
            batch_ids = question_ids[i:i+chunk_size]
            run_agents_and_catch_logs(
                model_name=model_name, instance_ids=batch_ids, instance_cost_limit=cost_limit, split=split, dataset_name=dataset_name
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
                full_dataset_name=dataset_name,
                cost_limit=cost_limit,
                temperature=0.00,
                top_p=0.95,
                run_id="test",
                split=split,
                max_workers=8,
                full_dataset=d,
                test_ids=question_ids,
            )

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

if __name__ == "__main__":
    #Next do SWE-bench 0-220
    main(
        dataset_name="princeton-nlp/SWE-bench",
        model_name="gpt4o",
        cost_limit=0.75,
        split="dev",
        first_question_index=0,
        last_question_index=100,
    )
    #ERROR: No matching distribution found for vtk, hd5py, pyvista
    
    # Lite Dev Sonnet (75 cents) 0-23
    # Succeeded pvlib__pvlib-python-1154
    # Progress pydicom__pydicom-1256, pydicom__pydicom-901, pylint-dev__astroid-1866

    # Lite Dev gpt-4o (75 cents) 0-23
    # Succeeded - pvlib__pvlib-python-1072, pydicom__pydicom-1694
    # Progress - marshmallow-code__marshmallow-1343, marshmallow-code__marshmallow-1359, pvlib__pvlib-python-1154, pydicom__pydicom-1139, pydicom__pydicom-901, pylint-dev__astroid-1196, pylint-dev__astroid-1268, pylint-dev__astroid-1866, pylint-dev__astroid-1978

    # Full Dev Sonnet (25 cents) 150-200
    # Progress: pydicom__pydicom-1000, pydicom__pydicom-1048, pydicom__pydicom-1228, pydicom__pydicom-1241, pydicom__pydicom-1050

    # Full Dev GPT-4o (75 cents) 0-100
    # Success: marshmallow-code__marshmallow-1343, vlib__pvlib-python-1026, pvlib__pvlib-python-1072, pvlib__pvlib-python-1093, pvlib__pvlib-python-1160, pvlib__pvlib-python-1191, pvlib__pvlib-python-1216, pvlib__pvlib-python-1273