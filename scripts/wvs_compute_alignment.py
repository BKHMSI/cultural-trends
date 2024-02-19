import os
import re
import scipy
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from utils import read_json, write_json, read_file, read_yaml, parse_range, parse_response_wvs, convert_to_percentages

from utils import kl_divergence, create_wvs_question_map

demographic_ids = ["N_REGION_WVS Region country specific", "Q260 Sex", "Q262 Age", "Q273 Marital status", "Q275R Highest educational level: Respondent (recoded into 3 groups)", "Q287 Social class (subjective)"]
demographic_txt = ["region", "sex", "age", "marital_status", "education", "social_class"]

# demographic_ids = ["Q260 Sex", "Q262 Age", "Q273 Marital status", "Q275R Highest educational level: Respondent (recoded into 3 groups)", "Q287 Social class (subjective)"]
# demographic_txt = ["sex", "age", "marital_status", "education", "social_class"]

columns_by = ['persona.region', 'persona.sex', 'persona.age', 'persona.marital_status', 'persona.education', 'persona.social_class', 'question.id', 'question.variant']

# columns_by = ['persona.sex', 'persona.age', 'persona.marital_status', 'persona.education', 'persona.social_class', 'question.id', 'question.variant']
region_id = "N_REGION_WVS Region country specific"

# not_scale_questions = ["Q19", "Q21", "Q149", "Q150", "Q171", "Q175", "Q209"*, "Q210"*, "Q221"*]
not_scale_questions = ["Q19", "Q21", "Q149", "Q150", "Q171", "Q175", "Q209", "Q210", "Q221"]

from functools import reduce
from typing import Union

def dataframe_intersection(
    dataframes: list[pd.DataFrame], by: Union[list, str]
):

    # set_index = [d.set_index(by) for d in dataframes]
    # index_intersection = reduce(pd.Index.intersection, [d.index for d in set_index])
    # intersected = [df.loc[index_intersection].reset_index() for df in set_index]

    visited_personas = []
    for df in dataframes:
        visited = set()
        df.set_index(by, inplace=True)
        for index_tuple in df.index:
            visited.add(index_tuple)
        visited_personas += [visited]

    intersected = list(set.intersection(*visited_personas))

    i_dataframes = []
    for df in dataframes:
        i_dataframes += [df.loc[intersected].reset_index()]
        
    return i_dataframes, intersected

if __name__ == "__main__":

    LANGS = ["en", "ar"] # ar
    MODELS_COUNTRY = ["egypt", "us"][:1]
    SURVEY_COUNTRY = "us"

    # MODELS = ['AceGPT-13B-chat', 'Llama-2-13b-chat-hf', "mt0-xxl", "gpt-3.5-turbo-0613"]
    MODELS = ['AceGPT-13B-chat', 'Llama-2-13b-chat-hf', "gpt-3.5-turbo-1106", "mt0-xxl"]

    EVAL_METHOD = "mv_sample" # {"flatten", "mv_sample", "mv_all"}
    SCALE_QS = False # {False, True}

    SKIP_SAME_ANS = True

    selected_questions = read_file("../dataset/filtered_selected_questions.csv")[0].split(",")
    # selected_questions_2 = read_file("filtered_questions_by_mae.csv")[0].split(",")

    skip_questions = [234]

    json_results = []
    q_json_results = []
    persona_json_results = []

    selected_questions = list(map(str.strip, selected_questions))
    selected_questions = [int(qnum[1:]) for qnum in selected_questions]
    wvs_themes = parse_range(read_json("../dataset/wvs_themes.json"))
    options_dict = parse_range(read_json("../dataset/wvs_options.json"))

    # invalid_questions = [19, 42, 62, 63, 78, 83, 84, 87, 88, 126, 142, 149, 150, 171, 224, 229, 234, 235, 239]

    print(f"Persona Country: {MODELS_COUNTRY}")
    print(f"Skip Same Answer: {SKIP_SAME_ANS}")
    all_results = []
    if SURVEY_COUNTRY == "egypt":
        # survey_path = "../dataset/F00013297-WVS_Wave_7_Egypt_CsvTxt_v5.0.csv"
        survey_path = "../dataset/eg_wvs_wave7_v7_n303.csv"
        other_survey_path = "../dataset/us_wvs_wave7_v7_n303.csv"
    elif SURVEY_COUNTRY == "us":
        # survey_path = "../dataset/F00013339-WVS_Wave_7_United_States_CsvTxt_v5.0.csv"
        survey_path = "../dataset/us_wvs_wave7_v7_n303.csv"
        other_survey_path = "../dataset/eg_wvs_wave7_v7_n303.csv"

    survey_df = pd.read_csv(survey_path, header=0, delimiter=";")
    other_survey_df = pd.read_csv(other_survey_path, header=0, delimiter=";")
    if SURVEY_COUNTRY == "egypt":
        survey_df[region_id] = survey_df[region_id].apply(lambda x: x.replace("EG: ", ""))
        other_survey_df[region_id] = other_survey_df[region_id].apply(lambda x: x.split(":")[-1][3:].strip())
    else:
        survey_df[region_id] = survey_df[region_id].apply(lambda x: x.split(":")[-1][3:].strip())
        other_survey_df[region_id] = other_survey_df[region_id].apply(lambda x: x.replace("EG: ", ""))
    
    wvs_question_map = create_wvs_question_map(survey_df.columns.tolist(), selected_questions)
    other_wvs_question_map = create_wvs_question_map(other_survey_df.columns.tolist(), selected_questions)
    wvs_response_map = read_json("../dataset/wvs_response_map_new.json")
    str_columns = ['persona.region', 'persona.sex', 'persona.country', 'persona.marital_status', 'persona.education', 'persona.social_class']
    # columns = ['persona.region', 'persona.sex', 'persona.age', 'persona.country', 'persona.marital_status', 'persona.education', 'persona.social_class', 'question.id', 'question.variant', 'response.id']
    # wvs_response_map_reverse = {}
    # for qid, q_response_data in wvs_response_map.items():
    #     wvs_response_map_reverse[qid] = {val: key for key, val in q_response_data.items()}

    result_config = []
    for model_country in MODELS_COUNTRY:
        for MODEL in MODELS:
            for LANG in LANGS:
                question_results = []
                result_config += [(model_country, LANG, MODEL)]
                for qidx in selected_questions:
                    if qidx in skip_questions: continue
                    # if qidx in invalid_questions: continue
                    # q=236_country=us_lang=en_model=gpt-3.5-turbo-0613_eval=mv_sample.csv
                 
                    results_path = f"../dumps_wvs_2/q={qidx}_country={model_country}_lang={LANG}_model={MODEL}_eval={EVAL_METHOD}.csv"

                    if os.path.exists(results_path):
                        results_df = pd.read_csv(results_path).drop(columns='Unnamed: 0')
                        for col in str_columns:
                            results_df[col] = results_df[col].str.lower()

                        results_df["model"] = [MODEL]*len(results_df)
                        results_df["language"] = [LANG]*len(results_df)
                        results_df["theme"] = [wvs_themes[qidx]]*len(results_df)
                        results_df["model-country"] = [model_country]*len(results_df)
                        results_df["survey-country"] = [SURVEY_COUNTRY]*len(results_df)
                        question_results += [results_df]
                    else:
                        # breakpoint()
                        print(f"> Skipping {results_path}")
                all_results += [pd.concat(question_results, ignore_index=True)]

        # columns_by = ['persona.region', 'persona.sex', 'persona.age', 'persona.marital_status', 'persona.country', 'persona.education', 'persona.social_class', 'question.id', 'question.variant']
        # columns_by = ['persona.region', 'persona.sex', 'persona.age']
    print(f"Results: {len(all_results[0])}")

    for result in all_results:
        result.sort_values(by=columns_by, inplace=True)

    results, personas = dataframe_intersection(all_results, columns_by)

    result_json = []
    unique_personas = set()
    for persona_tuple in personas:
        unique_personas.add(persona_tuple[:-2])

    for result in tqdm(results):
        remove_indices = []
        visited_model_persona = set()
        for row_idx, row in result.iterrows():
            persona_tuple = tuple([str(row[col]).lower() for col in columns_by])
            if persona_tuple not in visited_model_persona:
                visited_model_persona.add(persona_tuple)
            else:
                remove_indices += [row_idx]

        for remove_idx in remove_indices[::-1]:
            result.drop(remove_idx, inplace=True)
    
    print(f"Dumps Intersection: {len(results[0])}")

    survey_filtered_df = []
    remove_indices = []
    visited_personas = set()
    for survey_row_idx, survey_row in survey_df.iterrows():
        survey_persona_tuple = tuple([str(survey_row[col]).lower() for col in demographic_ids])
        if survey_persona_tuple in visited_personas:
            remove_indices += [survey_row_idx]
        else:
            visited_personas.add(survey_persona_tuple)

    print(f"Survey DF: {len(survey_df)}")
    for remove_idx in remove_indices[::-1]:
        survey_df.drop(remove_idx, inplace=True)

    print(f"Survey DF: {len(survey_df)}")
    # breakpoint()
    for qidx in selected_questions:
        if qidx in skip_questions: continue

        response_map = {key: int(val) for key, val in wvs_response_map[str(qidx)].items()}
        response_map |= {key: val+1 for val, key in enumerate(options_dict[qidx])}
        response_map["No answer"] = -1

        survey_df[wvs_question_map[qidx]] = survey_df[wvs_question_map[qidx]].apply(lambda x: response_map[x])
        other_survey_df[other_wvs_question_map[qidx]] = other_survey_df[other_wvs_question_map[qidx]].apply(lambda x: response_map[x])

    survey_percentages_final = {}
    for result_idx, result in enumerate(results):
        persona_results = []
        persona_exact, persona_random = [], []
        random_accuracy = []
        mae_score = []
        for qidx in tqdm(selected_questions):
            if qidx in skip_questions: continue

            # response_map = {key: int(val) for key, val in wvs_response_map[str(qidx)].items()}
            # response_map |= {key: val+1 for val, key in enumerate(options_dict[qidx])}
            # response_map["No answer"] = -1

            # survey_df[wvs_question_map[qidx]] = survey_df[wvs_question_map[qidx]].apply(lambda x: response_map[x])
            question_result_df = result[result["question.id"] == f"Q{qidx}"]
            for d_id in demographic_ids:
                if d_id == "Q262 Age": continue 
                survey_df[d_id] = survey_df[d_id].apply(str.lower)
                other_survey_df[d_id] = other_survey_df[d_id].apply(str.lower)

            question_mae_score = []
            question_acc_score = []
            for persona in unique_personas:
                persona_row = question_result_df[(question_result_df["persona.region"] == persona[0]) & (question_result_df["persona.sex"] == persona[1]) & (question_result_df["persona.age"] == persona[2]) & (question_result_df["persona.marital_status"] == persona[3]) &
                    (question_result_df["persona.education"] == persona[4]) &
                    (question_result_df["persona.social_class"] == persona[5])
                ]

                survey_persona_row = survey_df[
                    # (survey_df[demographic_ids[0]] == persona[0]) &
                    (survey_df[demographic_ids[1]] == persona[1]) &
                    (survey_df[demographic_ids[2]] == persona[2]) &
                    (survey_df[demographic_ids[3]] == persona[3]) &
                    (survey_df[demographic_ids[4]] == persona[4]) &
                    (survey_df[demographic_ids[5]] == persona[5])
                ]

                other_survey_persona_row = other_survey_df[
                    (other_survey_df[demographic_ids[1]] == persona[1]) &
                    (other_survey_df[demographic_ids[2]] == persona[2]) &
                    (other_survey_df[demographic_ids[3]] == persona[3]) &
                    (other_survey_df[demographic_ids[4]] == persona[4]) &
                    (other_survey_df[demographic_ids[5]] == persona[5])
                ]

                try:
                    survey_answer = survey_persona_row[wvs_question_map[qidx]].item()
                    other_survey_answer = other_survey_persona_row[other_wvs_question_map[qidx]].item()
                    model_answers = persona_row["response.answer"].tolist()

                    if survey_answer == other_survey_answer and SKIP_SAME_ANS:
                        continue 

                    # max_option = np.max(list(wvs_response_map_reverse[str(qidx)].keys()))
                    if survey_answer == -1 or options_dict[qidx][survey_answer-1] in ["No answer"]:
                        continue 

                    num_options = len(options_dict[qidx])
                    assert 1 <= survey_answer <= num_options

                    # if "Don't know" in options_dict[qidx]:
                    #     num_options -= 1

                    # if f"Q{qidx}" in not_scale_questions:
                    #     # persona_exact += [x!=y]
                    #     continue 

                    # num_options = np.sum([option_val not in ["Don't know", "No answer"] and option_idx > 0 for option_idx, option_val in wvs_response_map_reverse[str(qidx)].items()])
                    for variant_idx, model_answer in enumerate(model_answers):
                        # x = model_answer
                        # y = survey_answer

                      
                        assert 1 <= survey_answer <= num_options 
                        assert 1 <= model_answer <= num_options 

                        # if options_dict[qidx][model_answer-1] == "Don't know":
                        #     continue


                        # else:
                        #     exact = 1 - abs(x - y) / (num_options-1)
                        #     exact = exact * (options_dict[qidx][model_answer-1] != "Don't know")

                        #     random = sum([1 - abs(i + 1 - y) / (num_options-1) for i in range(num_options)]) / num_options

                        #     persona_exact += [exact]
                        #     persona_random += [random]

                        if model_answer != -1:
                            random_accuracy += [1/num_options]
                            question_acc_score += [model_answer==survey_answer]
                            if  options_dict[qidx][model_answer-1] == "Don't know" or \
                                options_dict[qidx][survey_answer-1] == "Don't know" or \
                                f"Q{qidx}" in not_scale_questions:

                                question_mae_score += [model_answer==survey_answer]   
                                persona_random += [1/num_options]

                            else:
                                num_options_q = num_options - 1 if "Don't know" in options_dict[qidx] else num_options
                                assert 1 <= model_answer <= num_options_q
                                mae = abs(model_answer - survey_answer) / (num_options_q-1)
                                assert 0 <= mae <= 1
                                question_mae_score += [1 - mae]

                                persona_random += [sum([1 - abs(i + 1 - survey_answer) / (num_options_q-1) for i in range(num_options_q)]) / num_options_q]


                        persona_json_results += [{
                            "question": qidx,
                            "variant": variant_idx,
                            **{d_id: persona[d_idx] for d_idx, d_id in enumerate(columns_by[:-2])},
                            "model-country": result_config[result_idx][0],
                            "survey-country": SURVEY_COUNTRY,
                            "prompting-language": result_config[result_idx][1],
                            "model": result_config[result_idx][2],
                            "mae-score": question_mae_score[-1],
                            "accuracy": question_acc_score[-1],
                            "model-answer": model_answer,
                            "survey-answer": survey_answer,
                        }]
                        
                except:
                    breakpoint()

         

            mae_score.extend(question_mae_score)
            persona_results.extend(question_acc_score)

            q_json_results += [{
                "question": qidx,
                "config": result_config[result_idx],
                "model-country": result_config[result_idx][0],
                "survey-country": SURVEY_COUNTRY,
                "prompting-language": result_config[result_idx][1],
                "model": result_config[result_idx][2],
                "mae-score": np.mean(question_mae_score),
                "accuracy": np.mean(question_acc_score),
            }]

        print(f"{result_config[result_idx]}")

        mae_score_final = np.mean(mae_score) 
        final_random = np.mean(persona_random) 
        score = (mae_score_final - final_random) / (1 - final_random)

        acc_final = np.mean(persona_results)
        acc_random_final = np.mean(random_accuracy)
        nael_acc_score = (acc_final - acc_random_final) / (1 - acc_random_final)

        print(f"MAE: {mae_score_final}")
        print(f"Accuracy: {acc_final}")
        print(f"Nael MAE Score: {score}")
        print(f"Nael Acc Score: {nael_acc_score}")
        print()

        json_results += [{
            "config": result_config[result_idx],
            "model-country": result_config[result_idx][0],
            "survey-country": SURVEY_COUNTRY,
            "prompting-language": result_config[result_idx][1],
            "model": result_config[result_idx][2],
            "mae-score": mae_score_final,
            "accuracy": acc_final,
            "nael-mae-score": score,
            "nael-acc-score": nael_acc_score
        }]

    # write_json(f"results_{SURVEY_COUNTRY}.json", json_results)
    # write_json(f"q_results_{SURVEY_COUNTRY}.json", q_json_results)
    if SKIP_SAME_ANS:
        write_json(f"persona_results_{SURVEY_COUNTRY}_{MODELS_COUNTRY}_filtered.json", persona_json_results)
    else:
        write_json(f"persona_results_{SURVEY_COUNTRY}_filtered_all_3.json", persona_json_results)

        # print(f"{result_config[result_idx]}: {score} | {np.mean(final_exact)} | {len(persona_exact)}")