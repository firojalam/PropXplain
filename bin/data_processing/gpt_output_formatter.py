import json


def reformat_jsonlines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()

            data = json.loads(line)

            # Get the text from the 'paragraph' or 'tweet' key (whichever is present)
            text = data.get('paragraph') or data.get('text') or ""

            # Get the list of predictions (could be empty)
            predictions = data.get('prediction', [])

            for pred_item in predictions:
                out_l = []

                for k,v in pred_item.items():
                    if k != "start" and k != "end" and k != "technique":
                        out_l.append(v)

                row_list = [text] + out_l

                # Write the row as tab-separated
                fout.write("\t".join(row_list) + "\n")

if __name__ == "__main__":
    o_in = "/Users/mhasanain/work/temp_propoganda/propaganda_detector/llm_output/final_full_annot_clean_bin_span_train_w_explain_gpt4o_out_p1.jsonl"
    o_out = "/Users/mhasanain/work/temp_propoganda/propaganda_detector/llm_output/final_full_annot_clean_bin_span_train_w_explain_gpt4o_out_p1_formatted.txt"
    reformat_jsonlines(o_in, o_out)

    o1_in = "/Users/mhasanain/work/temp_propoganda/propaganda_detector/llm_output/final_full_annot_clean_bin_span_train_w_explain_gpt4o1_out_p1.jsonl"
    o1_out = "/Users/mhasanain/work/temp_propoganda/propaganda_detector/llm_output/final_full_annot_clean_bin_span_train_w_explain_gpt4o1_out_p1_formatted.txt"
    reformat_jsonlines(o1_in, o1_out)