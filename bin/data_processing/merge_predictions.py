import json

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





if __name__ == "__main__":
    output_pred_file = '/llm_output/final_full_annot_clean_span_train_w_complete-exp_gpt4o1_p2_wNotProp_out.jsonl'
    input_pred_files = ['/Users/mhasanain/work/temp_propoganda/propaganda_detector/llm_output/final_full_annot_clean_span_train_w_complete-exp_gpt4o1_p2_wNotProp_out2.jsonl',
                        '/Users/mhasanain/work/temp_propoganda/propaganda_detector/llm_output/final_full_annot_clean_span_train_w_complete-exp_gpt4o1_p2_wNotProp_out3.jsonl',
                        '/Users/mhasanain/work/temp_propoganda/propaganda_detector/llm_output/final_full_annot_clean_span_train_w_complete-exp_gpt4o1_p2_wNotProp_out4.jsonl'
                        ]

    first_batch_data = read_results_data(output_pred_file)
    output_pred_data = {}

    for in_file in input_pred_files:
        data = read_results_data(in_file)
        for parid,row in data.items():
            if parid in first_batch_data: continue
            if parid in output_pred_data: continue

            output_pred_data[parid] = row

    with open(output_pred_file, 'a', encoding="utf-8") as output_file:
        for parid,row in output_pred_data.items():
            json_string = json.dumps(row, ensure_ascii=False)
            output_file.write(json_string + "\n")




