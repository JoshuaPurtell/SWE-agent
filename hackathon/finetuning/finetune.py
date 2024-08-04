import json
import os

# Currently, this file just contains a function that will create training dataset to finetune on.
# Generated file from `create_training_data` can be uploaded to OpenPipe console to finetune the model.
class Finetune():
    def __init__(self, instance_id):
        # Sample of how to instantiate openpipe client
        # Make sure you have the OPENAI_API_KEY and OPENPIPE_API_KEY set in your environment
        '''
        llmClient = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            openpipe = {
                "api_key": os.getenv("OPENPIPE_API_KEY"),
            }     
        )
        '''
        self.instance_id = instance_id
        self.input = None
        self.output = None

    def setInput(self, input):
        self.input = input

    def setOutput(self, output):
        self.output = output

    def transform_single_to_openai_chat_format(self, model = "gpt-4o-mini"):
        return {
            # only required field
            "messages" : [
                {
                    "role": "system",
                    # Can add additional context / summarized output here.
                    # As per OpenAI website: 
                    #   "You can define instructions 
                    #   in the user message, but the instructions set in the 
                    #   system message are more effective." 
                    "content": 
                    """You are a seasoned engineer who takes bug descriptions 
                    and generates a patch to fix the bug. """, 
                },
                {
                    "role": "user",
                    "content": self.input,
                },
                {
                    "role": "assistant",
                    "content": self.output, # consists of thought + action
                }
            ],
        }

    # Upload this output to OpenPipe on console
    def append_single_entry(self, model = "gpt-4o-mini"):
        if self.input is None or self.output is None:
            raise ValueError("Input and Output cannot be None")
        f = open("hackathon/finetuning/{example}_training_data.jsonl".format(example = model + "_" + self.instance_id), "a")
        f.write(str(self.transform_single_to_openai_chat_format()) + "\n")
        f.close()