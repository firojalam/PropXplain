import json
import os
import optparse
import requests
import ast
from dotenv import load_dotenv
import openai
import regex as re


from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type



def input_chat(input_prompt, api_url, model, headers):
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


def read_results_data(file_path):
    data = {}
    with open(file_path, 'r', encoding="utf-8") as json_file:
        for line in json_file:
            #print(line)
            result = json.loads(line.strip())
            if 'paragraph_id' in result:
                id = str(result["paragraph_id"])
            else:
                id = str(result['tweet_id'])
            data[id] = result

    print("Number of results loaded: {}".format(len(data)),flush=True)
    return data


def filter_data(id_list, data):
    print("Number of items before: {}".format(len(data)),flush=True)
    new_data = {}
    for data_id, row in data.items():
        if (data_id in id_list):
            continue
        else:
            new_data[data_id] = row

    print("Number of items after: {}".format(len(new_data)),flush=True)

    return new_data

def post_process_prop(response, row, final_labels):
    if not response.endswith("]"):
        response = response + "]"
    if not response.startswith("["):
        response = "[" + response

    response = re.sub(r"\s+", " ", response)

    response = response.replace(")]\n\n('full'","), ('full'")
    response = response.replace("[\n", "[")
    response = response.replace("(\n", "(")
    response = response.replace(")\n", ")")
    response = response.replace("),\n", "),")
    response = response.replace("\'\"", "\'“")

    preds_explains = ast.literal_eval(response)

    # print(preds_explains)

    new_preds = []
    for i, l in enumerate(final_labels):
        text = l['text']
        tec = l['technique']
        start = l['start']
        end = l['end']

        for j, pred_explain in enumerate(preds_explains):
            p = pred_explain[0]
            ex = pred_explain[1]
            if text == p and i == j:
                # Reformat the prediction from the labels and the new prediction and add the explanation
                new_preds.append({'text': text, 'technique': tec, 'start': start, 'end': end, 'explanation': ex})

    if len(new_preds) != len(final_labels):
        print(row['paragraph_id'], flush=True)
        print(final_labels, flush=True)
        print(preds_explains, flush=True)

    row['labels'] = new_preds
    # for l in row['prediction']: print(l)


    complete_explain = preds_explains[-1]
    # print(complete_explain)
    if complete_explain[0] == "full":
        row['full_explanation'] = complete_explain[1]

    return row


def post_process_notprop(response, row):
    response = response.replace("Response:","")
    response = response.strip()
    response = " ".join(response.split())
    response = re.sub(r"\s+", " ", response)

    row['full_explanation'] = response

    return row



@retry(wait=wait_random_exponential(min=3, max=120), stop=stop_after_attempt(5000),
       retry=retry_if_not_exception_type((openai.InvalidRequestError, openai.error.Timeout)))
def api_call(pars, output_file, err_output_file, api_url, model, headers, prompt_text_prop, prompt_text_notprop, no_rows):
    k = 0
    for parid, row in pars.items():
        if k == no_rows: break
        if k % 100 == 0 and k != 0: print("Done with %d rows so far..." % k,flush=True)

        if 'paragraph' in row:
            sentence = row['paragraph']
        else:
            sentence = row['text']

        final_labels = []
        prop_spans = []

        for label in row['labels']:
            #if label['text'] not in prop_spans:
            prop_spans.append((label['text'],label['technique']))
            final_labels.append(label)



        if len(prop_spans) == 0:
            input_prompt = prompt_text_notprop + "News piece: " + sentence + "\n" + \
                           "\n\nResponse: \n"

        else:
            input_prompt = prompt_text_prop + "News piece: " + sentence + "\n" + \
                           "Propaganda spans: " + str(prop_spans) + "\n\nResponse: \n"

        try:
            response = input_chat(input_prompt, api_url, model, headers)
            response = response.json()
            #print(response)

            if response["choices"][0]["finish_reason"] == "content_filter":
                new_preds = []
                # for i, l in enumerate(final_labels):
                #     text = l['text']
                #     tec = l['technique']
                #     start = l['start']
                #     end = l['end']
                #
                #     new_preds.append({'text': text, 'technique': tec, 'start': start, 'end': end, 'explanation': ""})

                json_string = json.dumps(row, ensure_ascii=False)
                err_output_file.write(json_string + "\n")
                err_output_file.flush()
                print("Paragraph filtered: " + parid, flush=True)
                continue

            else:
                response = response["choices"][0]["message"]["content"]
                if len(prop_spans) == 0:
                    post_process_notprop(response, row)
                else:
                    row = post_process_prop(response,row,final_labels)

            json_string = json.dumps(row, ensure_ascii=False)
            output_file.write(json_string + "\n")

        except Exception as e:
            row["error_msg"] = str(e)
            print(response,flush=True)
            print(parid)
            json_string = json.dumps(row, ensure_ascii=False)
            err_output_file.write(json_string + "\n")
            err_output_file.flush()

        k += 1


def continue_from_stopped(data, results_file_path):
    print("continuing from %s" % results_file_path,flush=True)

    id_list = set(read_results_data(results_file_path).keys())

    print("id_list: {}".format(len(id_list)),flush=True)

    data_subset = filter_data(id_list, data)

    return data_subset


def safe_open(path, option):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return open(path, option, encoding="utf-8")


def run_gpt_zero_shot(annots_fname, out_fname, err_output_file, api_url, model, headers, lang, no_rows):
    prompt_text1 = (
        f"You are provided with a news piece and a set of text spans that are subset of it and  "
        f" propagandistic. Each span is associated with a propaganda technique manifested in it. Generate a 1-sentence Explanation/rationale for each span, explaining why it is propagandistic."
        f"\nThen, inspect your generated explanations and generate one complete explanation shorter than 50 words on why the paragraph as a whole is propagandistic.\n "
        f"Your explanations must be fully in {lang} and your response should be formatted as a list of tuples [('Span 1','Explanation 1'), ('Span 2','Explanation 2'), etc], with "
        f" order of explanations matching order of spans. Append the complete explanation to the end of this list as a new tuple of this format: ('full', 'complete explanation')."
    )

    prompt_text2 = (
        f"You are provided with a news piece and a set of text spans that are subset of it and  "
        f" propagandistic. Each span is associated with a propaganda technique manifested in it. Generate a 1-sentence Explanation/rationale for each span, explaining why it is propagandistic."
        f"\nThen, inspect your generated explanations and generate one complete explanation shorter than 100 words on why the paragraph as a whole is propagandistic. Be very specific in this full explanation to the paragraph at hand.\n "
        f"Your explanations must be fully in {lang} and your response should be formatted as a list of tuples [('Span 1','Explanation 1'), ('Span 2','Explanation 2'), etc], with "
        f" order of explanations matching order of spans. Append the complete explanation to the end of this list as a new tuple of this format: ('full', 'complete explanation')."
    )

    prompt_text3 = (
        f"You are provided with a news piece and a set of text spans that are subset of it and  "
        f" propagandistic. Each span is associated with a propaganda technique manifested in it. Generate a 1-sentence Explanation/rationale for each span, explaining why it is propagandistic."
        f"\nThen, inspect your generated explanations and generate one complete explanation on why the paragraph as a whole is propagandistic. Be very specific in this full explanation to the paragraph at hand.\n "
        f"Your explanations must be fully in {lang} and your response should be formatted as a list of tuples [('Span 1','Explanation 1'), ('Span 2','Explanation 2'), etc], with "
        f" order of explanations matching order of spans. Append the complete explanation to the end of this list as a new tuple of this format: ('full', 'complete explanation')."
    )
    prompt_text4 = (
        f"You are provided with a news piece and a set of text spans that are subset of it and  "
        f" propagandistic. Each span is associated with a propaganda technique manifested in it. Generate a 1-sentence Explanation/rationale for each span, explaining why it is propagandistic."
        f"\nThen, inspect your generated explanations and generate one complete explanation shorter than 100 words on why the paragraph as a whole is propagandistic. Be very specific in this full explanation to the paragraph at hand.\n "
        f"Your explanations must be fully in {lang} and your response should be formatted as a list of tuples [('Span 1','Explanation 1'), ('Span 2','Explanation 2'), etc], with "
        f" order of explanations matching order of spans. Append the complete explanation to the end of this list as a new tuple of this format: ('full', 'complete explanation')."
    )

    extra_ar =  (f"The potential propaganda techniques that might appear in the text spans: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n"
              )

    extra_en =  (f"The potential propaganda techniques that might appear in the text spans: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Pity', 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n"
              )


    prompt_text_notprop = (
        f"You are provided with a news piece, generate one complete and concise explanation shorter than 100 words on why the paragraph as a whole is not propagandistic. Be very specific in this full explanation to the paragraph at hand.\n "
        f"Your explanation must be fully in {lang} and your response must be the explanation only without any extra text."
    )

    extra = ""
    #English has one more technique
    if lang == "Arabic":
        extra = extra_ar
    elif lang == "English":
        extra = extra_en

    prompt_text_prop = prompt_text2 + extra

    print('running on file: ' + str(annots_fname),flush=True)

    filtering_data = read_results_data(annots_fname)

    if os.path.isfile(out_fname):
        filtering_data = continue_from_stopped(filtering_data, out_fname)

    output_file = safe_open(out_fname, 'a')
    api_call(filtering_data, output_file, err_output_file, api_url, model, headers, prompt_text_prop, prompt_text_notprop, no_rows)

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
    parser.add_option('-l', '--lang', action='store', dest='l', default=None, type="string",
                      help='Two letter language code of samples')
    parser.add_option('-r', '--rows', action='store', dest='r', default=None, type="int",
                      help='Number of samples to run on')
    parser.add_option('-a', '--api', action='store', dest='api', default=None, type="string",
                      help='API used: openai or azure')

    options, args = parser.parse_args()

    input_file = options.input_file
    err_output_file = safe_open(options.err_out_fname, 'a')
    api = options.a

    load_dotenv(dotenv_path=options.env, override=True)

    model = os.environ['OPENAI_MODEL']
    openai_api_base = os.environ['OPENAI_API_BASE']
    openai_api_key = os.environ['OPENAI_API_KEY']

    if api == "openai":
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
    elif api == "azure":
        openai_api_version = os.environ['OPENAI_API_VERSION']
        api_url = f"{openai_api_base}/openai/deployments/{model}/chat/completions?api-version={openai_api_version}"
        headers = {"api-key": openai_api_key}

    setup = options.setup
    if options.l == "ar":
        l = "Arabic"
    elif options.l == "en":
        l = "English"

    n = options.n

    print("Running %d lines.."%n,flush=True)

    if setup == "zshot":
        run_gpt_zero_shot(input_file, options.out_fname, err_output_file, api_url, model, headers, l, n)