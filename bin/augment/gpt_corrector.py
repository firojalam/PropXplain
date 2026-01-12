import json
import os
import optparse
import openai
from dotenv import load_dotenv


from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type



def input_chat(input_prompt, model):
    system_prompt = "You are an expert analyst of Arabic news articles.\n\n"

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{input_prompt}"},
    ]

    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0,
        top_p=0.95,
        max_tokens=800,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response


def read_results_data(file_path):
    data = {}
    print(file_path)
    with open(file_path, 'r', encoding="utf-8") as json_file:
        for line in json_file:
            result = json.loads(line)
            id = str(result["paragraph_id"])
            data[id] = result

    print("Number of results loaded: {}".format(len(data)))
    return data


def filter_data(id_list, data):
    print("Number of items before: {}".format(len(data)))
    new_data = {}
    for data_id, row in data.items():
        if (data_id in id_list):
            continue
        else:
            new_data[data_id] = row

    print("Number of items after: {}".format(len(new_data)))

    return new_data



@retry(wait=wait_random_exponential(min=3, max=120), stop=stop_after_attempt(5000),
       retry=retry_if_not_exception_type((openai.InvalidRequestError, openai.error.Timeout)))
def api_call(pars, output_file, err_output_file, model, base_prompt, no_rows):
    k = 0
    for parid, row in pars.items():
        if k == no_rows: break
        if k % 100 == 0 and k != 0: print("Done with %d rows so far..." % k)
        #print(row)

        sentence = row['paragraph']
        prop_spans = [label['text'] for label in row['prediction']]

        print(sentence + "\t" + "\t".join([s for s in prop_spans]))

        if len(prop_spans) == 0:
            print("Skipping row with no propaganda")

            row['llm_correction'] = []

            json_string = json.dumps(row, ensure_ascii=False)
            output_file.write(json_string + "\n")

            continue # skip paragraphs with no propaganda

        input_prompt = base_prompt + "Paragraph: " + sentence + "\n" + \
                       "Propaganda spans: " + str(prop_spans) + "\n" + \
                       "Explanations: " + str(row['llm_response']) + "\n\nResponse: \n"

        #print(input_prompt)

        try:
            response = input_chat(input_prompt, model)

            if response["choices"][0]["finish_reason"] == "content_filter":
                row['llm_correction'] = []
            else:
                response = response["choices"][0]["message"]["content"]
                row['llm_correction'] = response
                #for l in row['llm_response']: print(l)

            json_string = json.dumps(row, ensure_ascii=False)
            output_file.write(json_string + "\n")

        except Exception as e:
            row["error_msg"] = str(e)
            json_string = json.dumps(row, ensure_ascii=False)
            err_output_file.write(json_string + "\n")
            err_output_file.flush()

        k += 1


def continue_from_stopped(data, results_file_path):
    print("continuing from %s" % results_file_path)

    id_list = set(read_results_data(results_file_path).keys())

    print("id_list: {}".format(len(id_list)))

    data_subset = filter_data(id_list, data)

    return data_subset


def safe_open(path, option):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return open(path, option, encoding="utf-8")


def run_gpt_zero_shot(annots_fname, out_fname, err_output_file, model, no_rows):
    prompt_text = (
        f"You are provided with a Paragraph, a set of text spans that are subset of the paragraph and written in propagandistic"
        f" tone and a sentence for each span, explaining why it is propagandistic."
        f" Your task is to re-phrase the paragraph to eliminate the propgandistic writing (only if the propganda spans don't fall within a quote) and your response should be formatted as the paragraph after fixing."
        f"\n"
    )

    extra =  (f"Some common propaganda techniques used and you can use in your explanations: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice' , 'no technique'\n"
              )

    #prompt_text = prompt_text + extra
    print('running on file: ' + str(annots_fname))

    filtering_data = read_results_data(annots_fname)

    if os.path.isfile(out_fname):
        filtering_data = continue_from_stopped(filtering_data, out_fname)

    output_file = safe_open(out_fname, 'a')
    api_call(filtering_data, output_file, err_output_file, model, prompt_text, no_rows)

    output_file.close()
    err_output_file.close()

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input_file', action="store", dest="input_file", default=None, type="string",
                      help='input annotations to explain file')
    parser.add_option('-o', '--output_file', action="store", dest="out_fname", default=None, type='string',
                      help="output file for model responses.")
    parser.add_option('-e', '--err_output_file', action='store', dest='err_out_fname', default=None, type="string",
                      help='output file for failed model responses.')
    parser.add_option('-s', '--setup', action='store', dest='setup', default="zhot", type="string",
                      help='Learning setup: zshot | 3shot')
    parser.add_option('-v', '--env', action='store', dest='env', default=None, type="string",
                      help='API key file')
    parser.add_option('-l', '--lines', action='store', dest='l', default=None, type="int",
                      help='Number of samples to run on')

    options, args = parser.parse_args()

    input_file = options.input_file
    err_output_file = safe_open(options.err_out_fname, 'a')

    # need to set openai api keys
    load_dotenv(options.env)
    openai.api_type = os.getenv('OPENAI_API_TYPE')
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_version = os.getenv('OPENAI_API_VERSION')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL')

    print("running model: " + str(model))
    setup = options.setup
    l = options.l

    print("Running %d lines.."%l)

    if setup == "zshot":
        run_gpt_zero_shot(input_file, options.out_fname, err_output_file, model, l)