import json
import os

def reformat_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()

            data = json.loads(line)

            doc_id = data.get('paragraph_id') or data.get('tweet_id')
            text = data.get('paragraph') or data.get('text')

            # Get the list of predictions (could be empty)
            predictions = data.get('labels')
            explan = data.get('full_explanation', "")

            if len(predictions) > 0:
                label = 'propagandistic'
            else:
                label = 'non-propagandistic'

            print(doc_id)
            output = "Label: "+ label + "\nExplanation: " + explan

            out_sample = {'sample_id': str(doc_id), 'input': text, 'output':output }
            json_string = json.dumps(out_sample, ensure_ascii=False)
            fout.write(json_string + "\n")


if __name__ == "__main__":
    base_dir = "/data/with_explain"
    langs = ["arabic", "english"]
    for lang in langs:
        input_dir = base_dir + "/" + lang + "/"
        output_dir  = base_dir + "/" + lang + "_formatted/"

        for infile in os.listdir(input_dir):
            if infile == ".DS_Store": continue
            print(infile)
            reformat_dataset(input_dir+infile, output_dir+infile)