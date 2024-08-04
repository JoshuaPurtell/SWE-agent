import json
import os
from typing import List, Dict
from dataclasses import dataclass
# Ensure we get diversity based on 'action'
# create, edit 
# split based on space and take first one in action
# Remove based on 'observation'
taboo = ["Traceback","CommandError","Your proposed edit has introduced"]

@dataclass
class Datum:
    question_success: bool
    model: str
    messages: List[Dict]
    action: str

def filter_messages(messages, trajectory):
    filtered_messages = []
    for message, trajectory_step in zip(messages, trajectory):
        observation = trajectory_step['observation']
        if any([t in observation for t in taboo]):
            continue
        action = trajectory_step['action'].split(" ")[0].strip("\n")
        filtered_messages.append(Datum(messages=message, action=action))
    return filtered_messages

def balance_dataset(paths):
    data = []
    datasets = []
    for path in paths:
        datasets.append(json.load(open(path)))
    for dataset in datasets:
        if "information" in dataset.keys() and "trajectory" in dataset["information"].keys():
            m = dataset["ft_messages"]
            traj = dataset["information"]["trajectory"]['trajectory']
            if len(m) != len(traj):
                continue
            filtered_messages = filter_messages(m, traj)
            data.extend(filtered_messages)
    return data

if __name__ == "__main__":
    successful_path = "hackathon/data/successful"
    failed_path = "hackathon/data/failed"
    paths = [successful_path, failed_path]
    filtered_data = balance_dataset([os.path.join(path, f) for path in paths for f in os.listdir(path)])
    print(len(filtered_data))
    actions = [d.action for d in filtered_data]
    from collections import Counter
    action_counts = Counter(actions)

    # Probably want to bias the dataset based on which
    # create the biggest problems for Llama3-70b
    for action, count in action_counts.items():
        print(f"Action: {action}, Count: {count}")
    
    ft_data = [d.messages for d in filtered_data][0:50]
    print("Len: ",len(ft_data))
    output_path = "hackathon/learning_algos/test_filtered_ft_data_train_open_pipe.jsonl"
    with open(output_path, "w") as f:
        for item in ft_data:
            json.dump(item, f)
            f.write("\n")
    ft_data = [d.messages for d in filtered_data][50:]
    print("Len: ",len(ft_data))
    output_path = "hackathon/learning_algos/test_filtered_ft_data_val.jsonl"
    with open(output_path, "w") as f:
        for item in ft_data:
            json.dump(item, f)
            f.write("\n")
    print(f"Data saved to {output_path}")
