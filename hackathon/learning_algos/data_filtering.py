import json
import os

filteredDirectory = "hackathon/data/successful/"

def filter_by_successful_trajectory(jsonFileName):
    jsonFilePath = filteredDirectory + jsonFileName
    with open(jsonFilePath) as f:
        data = json.load(f)
        f.close()

    filteredFilePath = filteredDirectory + "filtered/" + jsonFileName[:-1*len(".json")]+"_filtered.jsonl"
    filteredFile = open(filteredFilePath, 'a')
    
    finetuning_messages = data["ft_messages"]
    trajectories = data["information"]["trajectory"]["trajectory"]

    if len(finetuning_messages) != len(trajectories):
        print("Length of finetuning_messages and trajectories do not match for ", jsonFileName)
        return

    for i, trajectory in enumerate(trajectories):
        observation = trajectory["observation"]
        if observation.find("Traceback") < 0 and observation.find("CommandError") < 0 and observation.find("Your proposed edit has introduced new syntax error(s)") < 0:
            filteredFile.write(str(finetuning_messages[i]) + "\n")
    filteredFile.close()

def filter_directory():
    for filename in os.listdir(filteredDirectory):
        if filename.endswith(".json"):
            filter_by_successful_trajectory(filename)

if __name__ == "__main__":
    filter_directory()
