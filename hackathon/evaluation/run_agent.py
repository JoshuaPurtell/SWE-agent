from __future__ import annotations

import time

from hackathon.evaluation.evaluate import get_runnable_problems, run_agents_and_catch_logs, run_swebench_evaluation
from getpass import getuser

if __name__ == "__main__":
    from datasets import load_dataset

    d = load_dataset("princeton-nlp/SWE-bench_Lite")

    mode = ["mini", "sonnet", "L3.1-70b-Together", "L3.1-70b-Baseten", "L3.1-405b-Baseten", "L3.1-70b-Groq"][2]
    if mode == "mini":
        model_name = "gpt-4o-mini"
        cost_limit = 0.22
    elif mode == "sonnet":
        model_name = "claude-3-5-sonnet-20240620"
        cost_limit = 1.5
    elif mode == "L3.1-70b-Together":
        model_name = "L3.1-70b-Together"
        cost_limit = 0.50
    elif mode == "L3.1-70b-Baseten":
        model_name = "L3.1-70b-BaseTen"
        cost_limit = 1.0
    elif mode == "L3.1-405b-Baseten":
        model_name = "L3.1-405b-BaseTen"
        cost_limit = 1.0
    elif mode == "L3.1-70b-Groq":
        model_name = "L3.1-70b-Groq"
        cost_limit = 1.0
    run_agent = True
    evaluate_agent = True
    
    split = "test"
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
    # first_question_index = 5
    # last_question_index = 20
    # question_ids = [
    #     d[split][question_index]["instance_id"]
    #     for question_index in range(first_question_index, last_question_index)
    # ]

    runnable_problems_by_split = get_runnable_problems(
        f"trajectories/{getuser()}/{model_name}__SWE-bench_Lite__default__t-0.00__p-0.95__c-{cost_limit:.2f}__install-1"
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
            run_swebench_evaluation(
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
        print("Time taken to evaluate runs: ", time.time() - t0)
