import re
import os
import yaml
import json
import time
import requests
import numpy as np
import pandas as pd

import scipy.stats
from itertools import product
from collections import Counter

MAX_ATTEMPTS = 10

def retry_request(url, payload, headers):
    for i in range(MAX_ATTEMPTS):
        try:
            response = requests.post(url, data=json.dumps(
                payload), headers=headers, timeout=90)
            json_response = json.loads(response.content)
            if "error" in json_response:
                print(json_response)
                print(f"> Sleeping for {2 ** i}")
                time.sleep(2 ** i) 
            else:
                return json_response
        except:
            print(f"> Sleeping for {2 ** i}")
            time.sleep(2 ** i)  # exponential back off
    raise TimeoutError()

def convert_to_percentages(answers, options, answer_map=None, is_scale=False):
    answers_mapped = []
    for ans in answers:
        if ans == -1: continue
        if ans not in options and answer_map is not None:
            answers_mapped += [str(answer_map[ans])]
        elif not is_scale:
            answers_mapped += [options[ans-1]]
        else:
            answers_mapped += [ans]
        
    # Count the occurrences of each answer
    answer_counts = Counter(answers_mapped)
    # Calculate the total number of answers
    total_answers = len(answers)
    # Calculate the percentage for each unique answer and store it in a dictionary
    percentages = {answer: (count / total_answers) * 100 for answer, count in answer_counts.items()}
    labels = list(percentages.keys())
    values = [percentages[label] if label in percentages else 0 for label in labels]
    return options, values

def parse_range(data):
    data_dict = {}
    for q_range in data:
        if "-" in q_range:
            q_start, q_end = tuple(map(int, q_range.split("-")))
        else:
            q_start = q_end = int(q_range)

        for q_idx in range(q_start, q_end+1):
            data_dict[q_idx] = data[q_range]

    return data_dict

def cartesian_product(lists):
    return list(product(*lists))

def read_file(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = fin.readlines()
    return data 

def read_raw(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = fin.read()
    return data 

def read_json(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = json.load(fin)
    return data 

def read_yaml(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data 

def write_file(path, data):
    with open(path, 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(data))

def write_json(path, data):
    with open(path, 'w', encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance

def append_row(
    data,
    **cols,
):
    for k, v in cols.items():
        data[k].append(v)


def parse_response_wvs(response: str, question_options: list):
    response = response.lower().strip()
    pattern = r"\(\d+\)"
    match = re.search(pattern, response)
    if match:
        answer = int(match.group()[1:-1])
        return answer if 1 <= answer <= len(question_options) else -1
    else:
        # for option_idx, option in enumerate(question_options):
        #     if response == option.lower().strip():
        #         return option_idx+1
        # question_options = question_options[::-1]
        # for option_idx, option in enumerate(question_options):
        #     if response == option.lower().strip():
        #         return len(question_options)-option_idx
        for option_idx, option in enumerate(question_options):
            if response in option.lower().strip():
                return option_idx+1
            
    pattern = r"\(\d+"
    match = re.search(pattern, response)
    if match:
        answer = int(match.group()[1:])
        return answer if 1 <= answer <= len(question_options) else -1

    pattern = r"\d+"
    match = re.search(pattern, response)
    if match:
        answer = int(match.group())
        return answer if 1 <= answer <= len(question_options) else -1

    return -1

def parse_response(res: str, options: list):
    if type(res) == int:
        return res 
        
    res = res.strip()
    pattern = r"\d+"
    match = re.search(pattern, res)
    if match:
        answer = int(match.group())
        if 1 <= answer <= len(options):
            return answer 
    
    num_words = len(res.split())
    for i, option in enumerate(options):
        space_idx = option.index(" ")
        if res == option or \
           res == option.replace(".", "").strip() or \
           res == option[space_idx+1:].strip() or \
           res == option[space_idx+1:].strip().replace(".", "") or \
           res == ' '.join(option[:num_words]):  
            return i+1 
        
    for i in range(1, len(options)+1):
        if str(i) in res:
            return i
    return -1

def parse_question(q: dict, questions_en=None):
    index = '.'.join(str(x) for x in q['index'])
    text = q['questions'][0]

    if questions_en is not None:
        options = questions_en[index]["options"]
    else:
        options = q["options"]

    qparams = q["question_parameters"] if "question_parameters" in q else None
    
    return {
        'index': index,
        'text': text,
        'options': options,
        "qparams": qparams
    }

def append_data(qidx, data, questions, columnar_data):

    invalid_ans = 0
    for row in data:
        try:
            if "Error" in row:
                continue
            persona = row['persona']
            qid = '.'.join(str(x) for x in row['question']['id'])
            vid = row['question']['variant']
            responses = row['response']
            qparams = row["question"]["params"]
            key_qparam = list(qparams.keys())[0] if len(qparams) > 0 else None 
            if qidx == 6 and qparams[key_qparam] in ["Corporations", "Public Companies","Local Government", "Electoral Process"]:
                continue

            for response in responses:
               
                question = questions[qid]
                options = question["options"]
                answer = parse_response(response, options)
                if answer == -1:
                    invalid_ans += 1
                    continue 

                if key_qparam is not None:
                    qparam_idx = str(question["qparams"][key_qparam].index(qparams[key_qparam]) + 1)
                else:
                    qparam_idx = "0"

                if qidx == 10 and qparam_idx == "2":
                    # to remove the extra variant Nael added 
                    continue

                # breakpoint()

                append_row(
                    columnar_data,
                    qid=qid, vid=vid, response=answer,
                    question_text=question['text'],
                    response_text=question['options'][answer-1],
                    qparam_id=qparam_idx,
                    **persona,
                )
        except:
            breakpoint()
            raise
    
    print('='*50)
    print(f"> {invalid_ans} Invalid Answers")
    print('='*50)
    return columnar_data, invalid_ans

def read_question(path, qidx, questions_en=None):
    questions = {}
    with open(path, 'r', encoding='utf-8') as fp:
        q_data = yaml.safe_load(fp)['dataset']
        for row in q_data:
            if row["index"][0] != qidx: continue
            q = parse_question(row, questions_en)
            questions[q['index']] = q
    return questions

def get_results_path(filesuffix, model_name, lang, version, m1):
    for v_num in range(version, 0, -1):
        if m1:
            v_num = f"{v_num}m1"
        results_path = f'results/{model_name}/{lang}/preds_{filesuffix}_v{v_num}.json'
        if os.path.exists(results_path):
            return results_path
    return None 

def append_response(model_data:list[dict], 
        row:dict, 
        response_int:int, 
        response_id:int, 
        persona_id: int,
        q_responses:list[int]
    ):

    if q_responses is not None:
        most_common_responses = Counter(q_responses).most_common()
        max_freq = most_common_responses[0][1]
        max_responses = []
        for most_common in most_common_responses:
            if most_common[1] == max_freq:
                max_responses += [most_common[0]]
            else:
                break 
            
        response_int = np.random.choice(max_responses)

    assert response_int > 0
    model_data += [{
        "persona.region": row["persona"]["region"],
        "persona.sex": row["persona"]["sex"],
        "persona.age": row["persona"]["age"],
        "persona.country": row["persona"]["country"],
        "persona.marital_status": row["persona"]["marital_status"],
        "persona.education": row["persona"]["education"],
        "persona.social_class": row["persona"]["social_class"],
        "question.id": row["question"]["id"],
        "question.variant": row["question"]["variant"],
        "response.id": response_id,
        "response.answer": response_int
    }] 
    return model_data


def convert_to_dataframe(model_data:list[dict], 
                         question_options:list[str], 
                         demographic_map: dict[str,str], 
                         eval_method: str = "mv_all", 
                         language: str = "en",
                         is_scale_question: bool = False) -> pd.DataFrame:
    assert eval_method in {"flatten", "mv_sample", "mv_all", "first"}
    model_data_flat = []
    invalid_count = 0
    q_responses = []
    for row_idx, row in enumerate(model_data):

        if language != "en":
            row["persona"] = {d_text: (demographic_map[d_text][d_value] if d_text != "region" else demographic_map[d_text][d_value]) if d_text != "age" else d_value for d_text, d_value in row["persona"].items()}
        
        if eval_method == "mv_sample":
            q_responses = []

        if row_idx % 4 == 0 and eval_method == "mv_all":
            q_responses = []

        assert row_idx % 4 == row["question"]["variant"]

        for response_id, response in enumerate(row["response"]):
            response_int = parse_response_wvs(response, question_options)
            if is_scale_question:
                response_int = int(np.ceil(response_int/2))

            if eval_method == "first":
                if response_int <= 0: 
                    invalid_count += 1
                else:
                    model_data_flat = append_response(model_data_flat, row, response_int, response_id, row_idx, q_responses=None)
                break
       
            if eval_method == "flatten":
                if response_int <= 0: 
                    invalid_count += 1
                    continue
                model_data_flat = append_response(model_data_flat, row, response_int, response_id, row_idx, q_responses=None)
            elif response_int > 0 and "mv" in eval_method:
                q_responses += [response_int]

        if eval_method == "mv_sample":
            if len(q_responses) == 0:
                # breakpoint()
                invalid_count += 1 
                continue

            model_data_flat = append_response(model_data_flat, row, -1, response_id, row_idx, q_responses)

        elif eval_method == "mv_all" and row_idx % 4 == 3:
            if len(q_responses) == 0:
                invalid_count += 1 
                continue

            model_data_flat = append_response(model_data_flat, row, -1, response_id, row_idx, q_responses)

    return pd.DataFrame(model_data_flat), invalid_count

def create_wvs_question_map(headers:list[str], selected_questions:list[str]):
    wvs_question_map = {}
    for column in headers:
        match = re.search(r"Q(\d+)[\w]? (.+)", column)
        if match:
            qidx = int(match.group(1))
            if qidx in selected_questions:
                wvs_question_map[qidx] = column
    return wvs_question_map