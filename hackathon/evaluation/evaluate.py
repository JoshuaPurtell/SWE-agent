from __future__ import annotations

import io
import re
import subprocess
import time
from contextlib import redirect_stdout
from getpass import getuser
from typing import List

from run import (
    CONFIG_DIR,
    ActionsArguments,
    AgentArguments,
    EnvironmentArguments,
    ModelArguments,
    ScriptArguments,
    main,
)


def get_args_dev(
    model_name=None,
    instance_to_filter_by="marshmallow-code__marshmallow-1359",
    per_instance_cost_limit=0.025,
    split="dev",
    dataset_name="princeton-nlp/SWE-bench_Lite",
) -> ScriptArguments:
    return ScriptArguments(
        suffix="",
        environment=EnvironmentArguments(
            image_name="sweagent/swe-agent:latest",
            data_path=dataset_name,
            split=split,
            verbose=False,
            install_environment=True,
            cache_task_images=False,
        ),
        skip_existing=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name=model_name,
                total_cost_limit=0.0,
                per_instance_cost_limit=per_instance_cost_limit,
                temperature=0.0,
                top_p=0.95,
            ),
            config_file=CONFIG_DIR / "default.yaml",
        ),
        actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
        instance_filter=instance_to_filter_by,
    )


def get_runnable_problems(trajectory_path,dataset_name="princeton-nlp/SWE-bench_Lite"):
    import os

    from datasets import load_dataset

    d = load_dataset(dataset_name)

    traj_files = []
    for root, dirs, files in os.walk(trajectory_path):
        for file in files:
            if file.endswith(".traj"):
                traj_files.append(file.split(".")[0])
    dev_question_ids = [q["instance_id"] for q in d["dev"]]
    test_question_ids = [q["instance_id"] for q in d["test"]]

    return {
        "dev": [q for q in dev_question_ids if q in traj_files],
        "test": [q for q in test_question_ids if q in traj_files],
    }


def compare_filename_in_patches(pred_patch, expected_patch):
    if not pred_patch or expected_patch is None:
        return 0.0
    pred_match = re.findall(r"\+\+\+ b/(.*)", pred_patch)
    if not pred_match:
        return 0.0

    pred_filenames = {match.lower().strip() for match in pred_match}

    expected_match = re.findall(r"\+\+\+ b/(.*)", expected_patch)
    if not expected_match:
        return 1.0
    expected_filenames = {match.lower().strip() for match in expected_match}
    if not expected_filenames:
        return 0.0

    matched_filenames = pred_filenames & expected_filenames
    return len(matched_filenames) / len(expected_filenames) * 100.0


def run_swebench_evaluation(
    predictions_path_override=None,
    model_name=None,
    full_dataset_name="princeton-nlp/SWE-bench_Lite",
    cost_limit=0.05,
    temperature=0.00,
    top_p=0.95,
    max_workers=1,
    run_id="josh-testing",
    split="dev",
    dev_ids=None,
    test_ids=None,
    full_dataset=None,
):
    if predictions_path_override is None:
        dataset_name = full_dataset_name.split("/")[-1]
        predictions_path = f"trajectories/{getuser()}/{model_name}__{dataset_name}__default__t-{temperature:.2f}__p-0.95__c-{cost_limit:.2f}__install-1/all_preds.jsonl"
    else:
        predictions_path = predictions_path_override

    ids_by_split = get_runnable_problems("/".join(predictions_path.split("/")[:-1]),full_dataset_name)
    import json

    # Load all predictions
    print("Loading from: ",predictions_path)
    with open(predictions_path) as f:
        all_preds = [json.loads(line) for line in f]
    print("N all preds: ", len(all_preds))
    # Separate predictions into dev and test

    dev_ids = dev_ids if dev_ids is not None else ids_by_split["dev"]
    test_ids = test_ids if test_ids is not None else ids_by_split["test"]
    dev_preds = [pred for pred in all_preds if pred["instance_id"] in dev_ids]
    test_preds = [pred for pred in all_preds if pred["instance_id"] in test_ids]

    # Save dev predictions
    dev_preds_path = predictions_path.replace("all_preds.jsonl", "all_dev_preds.jsonl")
    with open(dev_preds_path, "w") as f:
        for pred in dev_preds:
            json.dump(pred, f)
            f.write("\n")

    # Save test predictions
    test_preds_path = predictions_path.replace("all_preds.jsonl", "all_test_preds.jsonl")
    with open(test_preds_path, "w") as f:
        for pred in test_preds:
            json.dump(pred, f)
            f.write("\n")

    # Update predictions_path to use the appropriate file based on the split
    predictions_path = dev_preds_path if split == "dev" else test_preds_path
    preds = dev_preds if split == "dev" else test_preds
    if len(preds) == 0:
        print(f"No predictions found for split {split}")
        return [], [], {}
    else:
        print(f"Running evaluation for split {split} - {len(preds)} predictions found")

    milestone_1_percents = {}
    traceback_counts = {}
    command_error_counts = {}
    syntax_error_counts = {}
    trajectories = {}

    dataset = full_dataset[split]

    preds = [
        pred for pred in preds if pred["instance_id"] in [example["instance_id"] for example in dataset]
    ]
    for pred in preds:
        instance_id = pred["instance_id"]
        traj_path = "/".join(predictions_path.split("/")[:-1]) + f"/{instance_id}.traj"
        #print("Traj path: ", traj_path)
        with open(traj_path) as f:
            traj = f.read()
        trajectories[instance_id] = json.loads(traj)
        traceback_count = traj.count("Traceback")
        command_error_count = traj.count("Command failed")
        syntax_error_count = traj.count("SyntaxError")
        #print(f"Instance {instance_id} - Tracebacks: {traceback_count}, Command Errors: {command_error_count}, Syntax Errors: {syntax_error_count}")
        
        traceback_counts[instance_id] = traceback_count
        command_error_counts[instance_id] = command_error_count
        syntax_error_counts[instance_id] = syntax_error_count

        filtered_dataset = dataset.filter(lambda example: example["instance_id"] == instance_id)
        expected = filtered_dataset[0]
        milestone_1_percents[instance_id] = compare_filename_in_patches(pred["model_patch"], expected["patch"])
    

    milestone_1_success = [id for id, percent in milestone_1_percents.items() if percent == 100]
    milestone_1_mean = sum(milestone_1_percents.values()) / len(milestone_1_percents) if milestone_1_percents else 0
    print(f"PATCHED ALL FILES: {len(milestone_1_success)}/{len(preds)}")
    
    command = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        full_dataset_name,
        "--predictions_path",
        predictions_path,
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--split",
        split,
    ]

    # Split out data from dev/test into separate preds file and run on it
    t0 = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    print("Time taken to run swebench: ", time.time() - t0)
    lines = result.stdout.split("\n")
    success_ids = []
    failed_ids = []
    for line in lines:
        if "Report written to " in line:
            print("Hit")
            file_name = line.replace("Report written to ", "")
            with open(file_name) as f:
                summary = json.load(f)
            failed_ids = list(set(summary["unresolved_ids"]).union(set(summary['error_ids'])).union(set(summary['empty_patch_ids'])))
            success_ids = summary["resolved_ids"]
            from colorama import Fore, Style, init

            init(autoreset=True)

            print("\nResults:")
            for id in success_ids + failed_ids:
                color = Fore.LIGHTGREEN_EX if id in success_ids else Fore.LIGHTRED_EX
                perc = milestone_1_percents.get(id, 0)
                milestone_1_emoji = "✅" if perc == 100 else "❌"
                milestone_1 = f" (PATCHED FILES: {perc}% {milestone_1_emoji})"
                print(f"{color}• {id}{milestone_1}{Style.RESET_ALL}")
    if len(success_ids) + len(failed_ids) == 0:
        print("No results found")
        return [], [], {}
    print(f"MEAN FILES PATCHED: {milestone_1_mean:.2f}%")
    print(f"SUCCESS RATE: {100 * len(success_ids) / (len(success_ids) + len(failed_ids)):.2f}%")

    instance_infos = {}
    metrics = [
        ("milestone_1_percent", milestone_1_percents),
        ("traceback_count", traceback_counts),
        ("command_error_count", command_error_counts),
        ("syntax_error_count", syntax_error_counts),
        ("trajectory", trajectories),
    ]

    for id in success_ids:
        instance_infos[id] = {
            "status": "success",
            **{name: metric.get(id, 0) for name, metric in metrics},
        }

    for id in failed_ids:
        instance_infos[id] = {
            "status": "failed",
            **{name: metric.get(id, 0) for name, metric in metrics},
        }
    return success_ids, failed_ids, instance_infos

def run_agent_and_catch_logs(
    model_name=None, instance="marshmallow-code__marshmallow-1359", cost_limit=0.05, split="dev", verbose=False, dataset_name="princeton-nlp/SWE-bench_Lite"
):
    if verbose:
        main(
            get_args_dev(
                model_name=model_name, instance_to_filter_by=instance, per_instance_cost_limit=cost_limit, split=split, dataset_name=dataset_name
            )
        )
        return []
    
    output = io.StringIO()
    with redirect_stdout(output):
        main(
            get_args_dev(
                model_name=model_name, instance_to_filter_by=instance, per_instance_cost_limit=cost_limit, split=split, dataset_name=dataset_name
            )
        )

    captured_logs = output.getvalue()
    log_lines = captured_logs.splitlines()
    print(f"Captured {len(log_lines)} lines of logs")
    return log_lines


def run_agents_and_catch_logs(model_name, instance_ids: List, instance_cost_limit: float, split: str, verbose=False, dataset_name="princeton-nlp/SWE-bench_Lite"):
    import multiprocessing

    num_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.starmap(
            run_agent_and_catch_logs,
            [(model_name, instance_id, instance_cost_limit, split, verbose, dataset_name) for instance_id in instance_ids],
        )

    log_lines = [line for result in results for line in result]
    return log_lines


# TODO: goal here is for us to be able to run swe-agent and then eval it with swe-bench to know correct/incorrect.
# Then, to enable us to add scoring functions that parse through the logged lines and keep track of intermediate metrics
