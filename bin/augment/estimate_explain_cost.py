import json

import tiktoken


def read_results_data(file_path):
    data = {}
    print(file_path)
    with open(file_path, 'r', encoding="utf-8") as json_file:
        for line in json_file:
            result = json.loads(line)
            if 'paragraph_id' in result:
                id = str(result["paragraph_id"])
            else:
                id = str(result['tweet_id'])
            data[id] = result

    print("Number of results loaded: {}".format(len(data)))
    return data

def count_all(base_dir,annots_fname):
    prompt_text = (
        f"You are provided with a news piece and a set of text spans that are subset of it and potentially "
        f" propagandistic. Generate a 1-sentence Explanation/rationale for each span, explaining why it is propagandistic."
        f" Your explanations must be fully in Arabic and your response should be formatted as a list of tuples [('Span 1','Explanation 1'), ('Span 2','Explanation 2'), etc], with "
        f" order of explanations matching order of spans.\n"
    )

    extra =  (f"Some common propaganda techniques that you can use in your explanations: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n"
              )

    prompt_text = prompt_text + extra
    print('running on file: ' + str(annots_fname))

    data = read_results_data(base_dir+annots_fname)

    total_tokens = 0
    rows_no_propaganda = 0

    for parid, row in data.items():
        k = 0
        if k % 100 == 0 and k != 0: print("Done with %d rows so far..." % k)

        if 'paragraph' in row:
            sentence = row['paragraph']
        else:
            sentence = row['text']

        prop_spans = []

        for label in row['labels']:
            if label['text'] not in prop_spans:
                prop_spans.append(label['text'])


        if len(prop_spans) == 0:
            rows_no_propaganda +=1
            #print("Skipping row with no propaganda")
            continue # skip paragraphs with no propaganda

        input_prompt = prompt_text + "News piece: " + sentence + "\n" + \
                       "Propaganda spans: " + str(prop_spans) + "\n\nResponse: \n"

        #print(input_prompt)
        total_tokens += count_tokens(input_prompt)

    print("Rows with no propaganda %d"%rows_no_propaganda)

    return total_tokens


def count_tokens(text, model='gpt-4o'):
    # Initialize the tokenizer for the specified model
    tokenizer = tiktoken.encoding_for_model(model)

    # Encode the text to get the list of tokens
    tokens = tokenizer.encode(text)

    # Return the number of tokens
    return len(tokens)


def count_all(base_dir,annots_fname):
    prompt_text = (
        f"You are provided with a news piece and a set of text spans that are subset of it and potentially "
        f" propagandistic. Generate a 1-sentence Explanation/rationale for each span, explaining why it is propagandistic."
        f" Your explanations must be fully in Arabic and your response should be formatted as a list of tuples [('Span 1','Explanation 1'), ('Span 2','Explanation 2'), etc], with "
        f" order of explanations matching order of spans.\n"
    )

    extra =  (f"Some common propaganda techniques that you can use in your explanations: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n"
              )

    prompt_text = prompt_text + extra
    print('running on file: ' + str(annots_fname))

    data = read_results_data(base_dir+annots_fname)

    total_tokens = 0
    rows_no_propaganda = 0

    for parid, row in data.items():
        k = 0
        if k % 100 == 0 and k != 0: print("Done with %d rows so far..." % k)

        if 'paragraph' in row:
            sentence = row['paragraph']
        else:
            sentence = row['text']

        prop_spans = []

        for label in row['labels']:
            if label['text'] not in prop_spans:
                prop_spans.append(label['text'])


        if len(prop_spans) == 0:
            rows_no_propaganda +=1
            #print("Skipping row with no propaganda")
            continue # skip paragraphs with no propaganda

        input_prompt = prompt_text + "News piece: " + sentence + "\n" + \
                       "Propaganda spans: " + str(prop_spans) + "\n\nResponse: \n"

        #print(input_prompt)
        total_tokens += count_tokens(input_prompt)

    print("Rows with no propaganda %d"%rows_no_propaganda)

    return total_tokens


def count_all_words(base_dir,annots_fname):
    data = read_results_data(base_dir+annots_fname)

    total_tokens = 0.0

    for parid, row in data.items():
        k = 0
        if k % 100 == 0 and k != 0: print("Done with %d rows so far..." % k)

        if 'paragraph' in row:
            sentence = row['paragraph']
        else:
            sentence = row['text']

        #print(input_prompt)
        print(count_words(sentence.strip()))
        total_tokens += count_words(sentence.strip())

    return total_tokens



def count_words(text):
    return len(text.split())


if __name__ == "__main__":
    base_dir = "/Users/mhasanain/work/temp_propoganda/data_annotation/final_full_annot/final_split/span/"
    annots_fnames = ["ArMPro_span_test_V2.jsonl", "final_full_annot_clean_span_dev.jsonl", "final_full_annot_clean_span_train.jsonl"]

    total_tokens = 0
    total_words = 0

    for annot_fname in annots_fnames:
        count = count_all(base_dir,annot_fname)
        print('Number of tokens for file %s is: %d'%(annot_fname,count))

        total_tokens += count

        count_words = count_all_words(base_dir,annot_fname)
        print('Number of words for file %s is: %d'%(annot_fname,count_words))

        total_words += count_words


    print('Total number of tokens is: %d' % (total_tokens))
    print('Total number of words is: %d' % (total_words))




