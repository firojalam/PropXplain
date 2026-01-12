from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import ast
import regex as re
import json


import os

# Load OpenAI API key
load_dotenv()
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize LLM
llm = AzureChatOpenAI(
    model="GPT-4o-2024-05-22", 
    openai_api_key=openai_api_key, 
    api_version='2024-02-15-preview',
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://tanbih.openai.azure.com/"),
    temperature=0.0
)

# Agent 1: Span Extractor
span_extractor_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are an expert in detecting propagandistic text spans. "
        "Given the following text, identify and extract all spans that exhibit propagandistic tendencies. If no spans are detected, return empty list [] directly."
        "Return each span as a list item.\n\n"
        "Input: {text}\n\n"
        "Return your response in this format:\n"
        "- Span 1: <span_text>\n- Span 2: <span_text>, etc.\n\n"
        "Output: \n"
    )
)

span_extractor_chain = LLMChain(llm=llm, prompt=span_extractor_prompt)

# Agent 2: Technique Classifier
technique_classifier_prompt = PromptTemplate(
    input_variables=["spans"],
    # template=(
    #     "You are an expert in classifying propagandistic techniques. "
    #     "Given the following spans with propagandistic tendencies, assign the appropriate propagandistic technique to each one and return a confidence score for each prediction. "
    #     "Choose from: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
    #     f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
    #     f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
    #     f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
    #     f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n\n"
    #     "Input: {spans}\n\n"
    #     "Use the following template for your response:\n"
    #     "- Span 1 (<Confidence score>): <span_text> → <technique>\n- Span 2 (<Confidence score>): <span_text> → <technique>, etc.\n\n"
    #     "Output: \n"
    # )
    template = (
        "You are an expert in classifying propagandistic techniques. "
        "Given the following spans with propagandistic tendencies, assign the appropriate propagandistic technique to each one. "
        "Choose from: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n\n"
        "Input: {spans}\n\n"
        "Use the following template for your response:\n"
        "- Span 1: <span_text> → <technique>\n- Span 2: <span_text> → <technique>, etc.\n\n"
        "Output: \n"
        )
)
technique_classifier_chain = LLMChain(llm=llm, prompt=technique_classifier_prompt)

# Agent 3: Verifier
verifier_prompt = PromptTemplate(
    input_variables=["classified_spans"],
    # template=(
    #     "You are an expert verifier for propaganda detection. "
    #     "You are provided with text spans that have propagandistic tendencies, and predicted propaganda techniques for them. Each prediction is associated with a confidence score."
    #     "Return a revised version with any corrections, or confirm that all classifications are correct.\n\n"
    #     "Input:\n{classified_spans}\n\n"
    #     "Use the following template for your response:\n"
    #     "- Span 1: <span_text> → <technique>\n- Span 2: <span_text> → <technique>, etc.\n\n"
    #     "Output: \n"
    # )
    template=(
        "You are an expert verifier for propaganda detection. "
        "You are provided with text spans that have propagandistic tendencies, and predicted propaganda techniques for them."
        "Return a revised version with any corrections, or confirm that all classifications are correct.\n\n"
        "Input:\n{classified_spans}\n\n"
        "Use the following template for your response:\n"
        "- Span 1: <span_text> → <technique>\n- Span 2: <span_text> → <technique>, etc.\n\n"
        "Output: \n"
    )
)
verifier_chain = LLMChain(llm=llm, prompt=verifier_prompt)

span_extractor_tool = Tool(
    name="Span Extractor",
    func=span_extractor_chain.run,
    description="Extract spans of text with propagandistic tendencies."
)

technique_classifier_tool = Tool(
    name="Technique Classifier",
    func=technique_classifier_chain.run,
    description="Classify spans of text with appropriate propagandistic techniques."
)

verifier_tool = Tool(
    name="Verifier",
    func=verifier_chain.run,
    description="Verify the correctness and coherence of classified spans."
)

tools = [span_extractor_tool, technique_classifier_tool, verifier_tool]

# Initialize the Coordinator Agent
coordinator_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)


def propaganda_detection_workflow(text):
    # Step 1: Extract spans
    #print("Step 1: Extracting spans...")
    prompt=(
        "You are an expert in detecting propagandistic text spans. If no spans are detected, return empty list [] directly."
        "Given the following text, identify and extract all spans that exhibit propagandistic tendencies. "
        "Return each span as a list item.\n\n"
        "Return your response in this format:\n"
        "- Span 1: <span_text>\n- Span 2: <span_text>, etc.\n\n"
    )

    span_extraction_result = coordinator_agent.run(prompt + f" {text}")

    if len(span_extraction_result) <5 or 'none' in span_extraction_result.lower() or 'no propagandistic' in span_extraction_result.lower() or 'does not contain' in span_extraction_result.lower():
        return []

    # Step 2: Classify techniques
    #print("\nStep 2: Classifying techniques...")
    # prompt = "You are an expert in classifying propagandistic techniques. " \
    #          "Given the following spans with propagandistic tendencies, assign the appropriate propagandistic technique to each one " \
    #          "and return a confidence score for each prediction. Choose from: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , " \
    #          "'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "\
    #         f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "\
    #         f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "\
    #         f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n\n"\
    #         "Use the following template for your response:\n"\
    #         "- Span 1 (<Confidence score>): <span_text> → <technique>\n- Span 2 (<Confidence score>): <span_text> → <technique>, etc.\n\n"\

    prompt = "You are an expert in classifying propagandistic techniques. " \
             "Given the following spans with propagandistic tendencies, assign the appropriate propagandistic technique to each one" \
             ". Choose from: 'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , " \
             "'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "\
            f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "\
            f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "\
            f"'Consequential Oversimplification' , 'False Dilemma/No Choice'\n\n"\
            "Use the following template for your response:\n"\
            "- Span 1: <span_text> → <technique>\n- Span 2: <span_text> → <technique>, etc.\n\n"\


    technique_classification_result = coordinator_agent.run(prompt + f" {span_extraction_result}")

    if len(technique_classification_result) <5 or 'none' in technique_classification_result.lower() or 'no propagandistic' in technique_classification_result.lower() or 'does not contain' in technique_classification_result.lower():
        return []

    # Step 3: Verify results
    #print("\nStep 3: Verifying results...")
    # prompt=(
    #     "You are an expert verifier for propaganda detection. "
    #     "You are provided with text spans that have propagandistic tendencies, and predicted propaganda techniques for them. Each prediction is associated with a confidence score."
    #     "Return a revised version with any corrections, or confirm that all classifications are correct.\n\n"
    #     "Return the spans in a list of tuple using this format:\n"
    #     "[(<span_text>,<technique>),(<span_text>,<technique>), etc]\n\n"
    # )
    prompt=(
        "You are an expert verifier for propaganda detection. "
        "You are provided with text spans that have propagandistic tendencies, and predicted propaganda techniques for them. " 
        "Return a revised version with any corrections, or confirm that all classifications are correct.\n\n"
        "Return the spans in a list of tuple using this format:\n"
        "[(<span_text>,<technique>),(<span_text>,<technique>), etc]\n\n"
    )
    verified_result = coordinator_agent.run(prompt + f" {technique_classification_result}")

    return verified_result

def find_span(span,par):
    try:
        # get the first matching span
        for match in re.finditer(span, par):
            start = match.start()
            end = match.end()
            break
    except:
        start = 0
        end = len(par)-1

    return start,end


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

def api_call(pars, output_file, err_output_file, no_rows):
    k = 0
    for parid, row in pars.items():
        if k == no_rows: break
        if k % 100 == 0 and k != 0: print("Done with %d rows so far..." % k)

        if 'paragraph' in row:
            sentence = row['paragraph']
        else:
            sentence = row['text']

        try:
            # Call agent workflow
            response = propaganda_detection_workflow(sentence)
            preds = []
            #print("================= Response ================")
            #print(row['paragraph_id'])
            #print(response)

            if len(response) == 0:
                row['prediction'] = []
            elif 'content filter' in response:
                row['prediction'] = []
            else:
                try:
                    preds = ast.literal_eval(response)
                except:
                    temp_preds = [s.replace("[","").replace("(","").replace("]","").replace(")","") for s in response.split("), (")]
                    for p in temp_preds:
                        s = str(p).split(",")
                        preds.append((s[0].strip(),s[1].strip()))

                new_preds = []
                for l in preds:
                    text = l[0]
                    tec = fix_single_label(l[1])
                    if tec == "no_technique": continue
                    start,end = find_span(text,sentence)

                    new_preds.append({'text':text,'technique':tec,'start':start,'end':end})

                row['prediction'] = new_preds

            json_string = json.dumps(row, ensure_ascii=False)
            output_file.write(json_string + "\n")

        except Exception as e:
            row["error_msg"] = str(e)
            json_string = json.dumps(row, ensure_ascii=False)
            err_output_file.write(json_string + "\n")
            err_output_file.flush()

        k += 1



def fix_single_label(label):
    label = label.strip().lower()
    if "slogan" in label:
        label_fixed = "Slogans"
    if "loaded" in label:
        label_fixed = "Loaded_Language"
    if "prejudice" in label or "fear" in label or "mongering" in label:
        label_fixed = "Appeal_to_Fear-Prejudice"
    if (
            "terminating" in label
            or "thought" in label
            or "conversation" in label
            or "killer" in label
    ):
        label_fixed = "Conversation_Killer"
    if "calling" in label or label == "name c" or "labeling" in label:
        label_fixed = "Name_Calling-Labeling"
    if (
            "minimisation" in label
            or label == "exaggeration minim"
            or "exaggeration" in label
    ):
        label_fixed = "Exaggeration-Minimisation"
    if "values" in label:
        label_fixed = "Appeal_to_Values"
    if "flag" in label or "wav" in label:
        label_fixed = "Flag_Waving"
    if "obfusc" in label or "vague" in label or "confusion" in label:
        label_fixed = "Obfuscation-Vagueness-Confusion"
    if "causal" in label:
        label_fixed = "Causal_Oversimplification"
    if "conseq" in label:
        label_fixed = "Consequential_Oversimplification"
    if "authority" in label:
        label_fixed = "Appeal_to_Authority"
    if "choice" in label or "dilemma" in label or "false" in label:
        label_fixed = "False_Dilemma-No_Choice"
    if "herring" in label or "irrelevant" in label:
        label_fixed = "Red_Herring"
    if "straw" in label or "misrepresentation" in label:
        label_fixed = "Straw_Man"
    if "guilt" in label or "association" in label:
        label_fixed = "Guilt_by_Association"
    if "questioning" in label or "reputation" in label:
        label_fixed = "Questioning_the_Reputation"
    if "whataboutism" in label:
        label_fixed = "Whataboutism"
    if "doubt" in label:
        label_fixed = "Doubt"
    if "doubt" in label:
        label_fixed = "Doubt"
    if "time" in label:
        label_fixed = "Appeal_to_Time"
    if "popularity" in label:
        label_fixed = "Appeal_to_Popularity"
    if "repetition" in label:
        label_fixed = "Repetition"
    if "hypocrisy" in label:
        label_fixed = "Appeal_to_Hypocrisy"

    if (
            "no propaganda" in label
            or "no technique" in label
            or label == ""
            or label == "no"
            or label == "appeal to history"
            or label == "appeal to emotion"
            or label == "appeal to"
            or label == "appeal"
            or label == "appeal to author"
            or label == "emotional appeal"
            or "no techn" in label
            or "hashtag" in label
            or "theory" in label
            or "specific mention" in label
            or "sarcasm" in label
            or "frustration" in label
            or "analogy" in label
            or "metaphor" in label
            or "religious" in label
            or "gratitude" in label
            or 'no_technique' in label
            or "technique" in label
            or 'rhetorical' in label):
        label_fixed = "no_technique"

    return label_fixed


def run_agent_zero_shot(annots_fname, out_fname, err_output_file, no_rows):
    print('running on file: ' + str(annots_fname))

    filtering_data = read_results_data(annots_fname)

    if os.path.isfile(out_fname):
        filtering_data = continue_from_stopped(filtering_data, out_fname)

    output_file = safe_open(out_fname, 'a')
    api_call(filtering_data, output_file, err_output_file, no_rows)

    output_file.close()
    err_output_file.close()


if __name__ == "__main__":
    #input_text = (
        #"المدينة نيوز :- ضمن سلسلة نجاحاتها المتتالية مؤخراً على المستوى الإقليمي والعالمي حققت جامعة عمان العربية / كلية علوم الطيران إنجازاً جديداً يضاف إلى الانجازات المتتالية في مجال المشاريع الدولية وذلك بعد أن نجحت في الحصول على دعم لمشروع Capacity Building in the field of Vocational Education and Training(CB-VET).")

    #result = propaganda_detection_workflow(input_text)

    #print("\nFinal Result:")
    #print(ast.literal_eval(result))

    base_dir = "/Users/mhasanain/work/temp_propoganda/propaganda_detector/"
    annots_fname = base_dir + "data/arabic/span/ArMPro_span_test_V2_100.jsonl"
    out_fname = base_dir + "llm_output/preds_TSD=ArMPro_span_test_V2_3steps_agent_gpt4o_out_100_noConf.jsonl"
    err_output_file = base_dir + "llm_output/preds_TSD=ArMPro_span_test_V2_3steps_agent_gpt4o_err_100_noConf.jsonl"
    err_output_file = safe_open(err_output_file, 'a')

    no_rows = 100
    run_agent_zero_shot(annots_fname, out_fname, err_output_file, no_rows)

