#!/usr/bin/env python
# coding: utf-8

import os
import json
import requests
import ast
import regex as re
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type


import anthropic


class AnthropicFailure(Exception):
    """Exception class to map various failure types from the AzureModel server"""

    def __init__(self, failure_type, failure_message):
        self.type_mapping = {
            "data_processing": "Model Inference failure",
            "connection": "Failed to connect to the API endpoint",
        }
        self.type = failure_type
        self.failure_message = failure_message

    def __str__(self):
        return (
            f"{self.type_mapping.get(self.type, self.type)}: \n {self.failure_message}"
        )



env_path = "/Users/mhasanain/work/Projects/LlamaLens/envs/claude-3-5-sonnet.env"
with open(env_path, 'r') as fh:
    vars_dict = dict(
        tuple(line.replace('\n', '').split('='))
        for line in fh.readlines() if not line.startswith('#')
    )


model_params = {}
ANTHROPIC_API_KEY = vars_dict['ANTHROPIC_API_KEY']
model_name = vars_dict['model_name']
model_params["top_p"] = 0.95
model_params["max_tokens"] = 4000
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@retry(wait=wait_random_exponential(min=3, max=120), stop=stop_after_attempt(5000),
       retry=retry_if_not_exception_type((TimeoutError, AnthropicFailure)))
def generate_claude(user_prompt, sys_prompt, temp):
    model_params["temperature"] = temp

    model_params["system"] = (
        sys_prompt
    )
    messages = [
        {"role": "user", "content": f"{user_prompt}"},
    ]

    response = client.messages.create(model=model_name, messages=messages, **model_params)

    response = json.loads(response.json())

    return response


def generate_gpt(input_prompt, api_url, model, headers):
    json_data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_prompt
                    }
                ]
            }
        ],
        # "temperature":temp,
        "top_p": 1,
        # "max_tokens": 4000,
        #       "response_format":{
        #         "type": "json_object",
        #       },
    }

    response = requests.post(api_url, headers=headers, json=json_data)

    return response





def process_reponse(response,lang):
    print(response)
    response = response.strip()

    response = response.replace('[```python ',"")
    response = response.replace('[```json ',"")
    response = response.replace('```]', "")
    response = response.replace('```python',"")
    response = response.replace('```json',"")

    response = response.replace('```',"")
    response = response.replace("[\n", "[")
    response = response.replace("]\n", "]")
    response = response.replace("\n[", "[")
    response = response.replace("\n]", "]")
    print(response)

    if not response.endswith("]"):
        response = response + "]"
    if not response.startswith("["):
        response = "[" + response


    response = re.sub(r"\s+", " ", response)

    response_text = response.replace("\n", " ").strip()
    response_text = response_text.replace("1.", "").replace("] 2. [", ",").strip()

    response_text = response_text.replace("[Here's a list of 10 diverse and concise English instructions for the propaganda detection task:","")
    response_text = response_text.replace("Here's a list of 10 diverse and concise English instructions for the propaganda detection task:","")

    response_text = response_text.strip()


    i = 1
    while i < 11:
        response_text = response_text.replace(str(i) + ".", ",")
        i += 1

    response_text = response_text.strip()


    response_text = response_text.replace("[[","[")

    response_text = response_text.strip()

    instructions = ast.literal_eval(response_text)

    #print(instructions)

    final_instructions = []
    for inst in instructions:
        if type(inst) == list:
            inst = inst[0]
            print(inst)
        #inst = inst.replace("[","").replace("]","").strip()


        out_inst = inst + " Return only the label and "+lang+" explanation in the following format:\nLabel: predicted label\nExplanation: "+lang+" explanation of predicted label."
        final_instructions.append(out_inst)
    print(final_instructions)
    print("\n")

    return final_instructions

if __name__ == "__main__":
    # got task info from
    #/Users/mhasanain/work/Projects/LlamaLens/support_data/datasets_tasks_labels_map.tsv
    env_path = '/Users/mhasanain/work/temp_propoganda/propaganda_detector/envs/gpt4o_openai.env'
    with open(env_path, 'r') as fh:
        vars_dict = dict(tuple(line.replace('\n', '').split('=')) for line in fh.readlines() if not line.startswith('#'))

    os.environ.update(vars_dict)
    openai_api_key = vars_dict['OPENAI_API_KEY']
    gpt_model = vars_dict['OPENAI_MODEL']

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }


    file_path = '/Users/mhasanain/work/temp_propoganda/propaganda_detector/data/instructions/instructions_gpt-4o_claude-3-5-sonnet_'

    temp = 0
    system_prompt = (
        "You are an expert LLM developer with expertise in writing instructions to instruction-tune LLMs for users' tasks. "
    )

    languages = [("Arabic","ar"), ("English", "en")]
    for lang in languages:
        out_file_path = file_path + lang[1] + ".json"
        user_prompt = (
            f"We are creating an English instruction-following dataset for an {lang[1]} "
            f"dataset covering the task of propaganda detection with explanation. The user defined the task as follows: "
            f"Detecting propaganda in a piece of text and explaining why this piece of text is propagandistic. Propaganda can be defined as a form of communication aimed at influencing people’s opinions or actions toward a specific goal, using well-defined rhetorical and psychological techniques. "
            f"For that task, the labels include: ['non-propagandistic', 'propagandistic']. "
            "Write 10 very diverse and concise English instructions making sure the labels provided above are part of the instruction. Only return the instructions without additional text. "
            "Return the instructions as strings in a list format as follows.\n[]"
        )



        response = generate_gpt(user_prompt, api_url,  gpt_model, headers)
        response = response.json()
        response = response["choices"][0]["message"]["content"]
        gpt_instructions = process_reponse(response,lang[0])

        response = generate_claude(user_prompt, system_prompt, temp)
        response = response["content"][0]["text"]
        claude_instructions = process_reponse(response,lang[0])

        all_instructions_ar = {}
        all_instructions_ar["gpt-4o"] = gpt_instructions
        all_instructions_ar["claude"] = claude_instructions


        with open(out_file_path, 'w') as file:
            json.dump(all_instructions_ar, file, ensure_ascii=False, indent=4)
