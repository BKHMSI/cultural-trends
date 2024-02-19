import os
import re
import scipy
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from utils import read_json, read_file, read_yaml, parse_range, parse_response_wvs, convert_to_percentages

from wvs_measure_distance import convert_to_dataframe, create_wvs_question_map

demographic_ids = ["N_REGION_WVS Region country specific", "Q260 Sex", "Q262 Age", "Q266 Country of birth: Respondent", "Q273 Marital status", "Q275R Highest educational level: Respondent (recoded into 3 groups)", "Q287 Social class (subjective)"]
demographic_txt = ["region", "sex", "age", "country", "marital_status", "education", "social_class"]

if __name__ == "__main__":

    COUNTRY = "egypt" # {egypt, us}
    LANGS = ["ar", "en"]

    # MODELS = ['AceGPT-13B-chat', 'Llama-2-13b-chat-hf', "gpt-3.5-turbo-0613", "mt0-xxl"]
    MODELS = ['AceGPT-13B-chat', 'Llama-2-13b-chat-hf', "gpt-3.5-turbo-1106", "mt0-xxl"]
    EVAL_METHOD = "mv_sample" # {"flatten", "mv_sample", "mv_all", "first"}
    SCALE_QS = False # {False, True}

    selected_questions = read_file("../dataset/filtered_selected_questions.csv")[0].split(",")
    selected_questions = list(map(str.strip, selected_questions))
    selected_questions = [int(qnum[1:]) for qnum in selected_questions]

    invalid_question_indices = []
    for LANG in LANGS:
        for MODEL in MODELS:
            print(f"######### {MODEL} #########\n")
            country_cap = "US" if COUNTRY == "us" else "Egypt"

            demographic_map = {}
            if LANG != "en":
                print("> Building Demographic Map")
                ar_persona_parameters = read_yaml(f"../dataset/wvs_template.ar.yml")["persona_parameters"]
                en_template_data = read_yaml(f"../dataset/wvs_template.en.yml")
                for d_text in demographic_txt:
                    if d_text == "age": continue
                    d_text_cap = ' '.join(list(map(str.capitalize, d_text.replace("_", " ").split())))
                    if d_text == "region":
                        d_values = en_template_data["persona_parameters"][d_text_cap][country_cap]
                    else:
                        d_values = en_template_data["persona_parameters"][d_text_cap]

                    demographic_map[d_text] = {}
                    for d_val_idx, d_val in enumerate(d_values):
                        if d_text == "region":
                            ar_d_val = ar_persona_parameters[d_text_cap][country_cap][d_val_idx]
                        else:
                            ar_d_val = ar_persona_parameters[d_text_cap][d_val_idx]

                        demographic_map[d_text][ar_d_val] = d_val
            
            if COUNTRY == "egypt":
                path = "../dataset/F00013297-WVS_Wave_7_Egypt_CsvTxt_v5.0.csv"
                path = "../dataset/eg_wvs_wave7_v7_n303.csv"
            elif COUNTRY == "us":
                path = "../dataset/us_wvs_wave7_v7_n303.csv"
                # path = "../dataset/F00013339-WVS_Wave_7_United_States_CsvTxt_v5.0.csv"

            survey_df = pd.read_csv(path, header=0, delimiter=";")
            if COUNTRY == "egypt":
                survey_df[demographic_ids[demographic_txt.index("region")]] = survey_df[demographic_ids[demographic_txt.index("region")]].apply(lambda x: x.replace("EG: ", ""))
            else:
                survey_df[demographic_ids[demographic_txt.index("region")]] = survey_df[demographic_ids[demographic_txt.index("region")]].apply(lambda x: x.split(":")[-1][3:].strip())

            wvs_question_map = create_wvs_question_map(survey_df.columns.tolist(), selected_questions)
            wvs_response_map = read_json("../dataset/wvs_response_map_new.json")
            
            if MODEL == "gpt-3.5-turbo-1106":
                dirpath = f"../results_wvs_2_gpt/{MODEL}/{LANG}"
            else:
                dirpath = f"../results_wvs_2/{MODEL}/{LANG}"

            options_dict = parse_range(read_json("../dataset/wvs_options.json"))
            wvs_themes = parse_range(read_json("../dataset/wvs_themes.json"))
            wvs_scale_questions = parse_range(read_json("../dataset/wvs_scale_questions.json"))

            wvs_questions = read_json(f"../dataset/wvs_questions_dump.{LANG}.json") 

            version_num = 3 if LANG == "en" and COUNTRY == "egypt" and MODEL in {"gpt-3.5-turbo-0613", "mt0-xxl"} else 1
            # if COUNTRY == "us":
            #     if "Llama-2" in MODEL:#å or "AceGPT" in MODEL:
            #         filepaths = sorted(glob(os.path.join(dirpath, f"*_country={COUNTRY}_*_maxt=32_n=5_v{version_num}_fewshot=0.json")))
            #     else:
            #         filepaths = sorted(glob(os.path.join(dirpath, f"*_country={COUNTRY}_*_v{version_num}*.json")))
            # else:
            #     if "Llama-2" in MODEL or "AceGPT" in MODEL:
            #         if LANG == "en" and 'Llama-2' in MODEL:
            #             filepaths = sorted(glob(os.path.join(dirpath, f"*maxt=32_n=5_v{version_num}_fewshot=0.json")))
            #         else:
            #             filepaths = sorted(glob(os.path.join(dirpath, f"*_v{version_num}_fewshot=0.json")))
            #     else:
            #         filepaths = sorted(glob(os.path.join(dirpath, f"*_v{version_num}.json")))

                # ar_filepaths = sorted(glob(os.path.join(f"../results_wvs/{MODEL}/ar", "*_v1.json")))

            if LANG == "ar" and MODEL == "AceGPT-13B-chat":
                filepaths = sorted(glob(f"{dirpath}/*_country={COUNTRY}_*_v2_*"))
            else:
                filepaths = sorted(glob(f"{dirpath}/*_country={COUNTRY}_*"))
                
            results = {demographic: [] for demographic in demographic_txt}
            print(f"> {len(filepaths)} Files")

            all_invalids, all_num_responses = 0, 0
            for filepath in tqdm(filepaths):
                if "country=us" in filepath and COUNTRY == "egypt":
                    continue

                if "country=egypt" in filepath and COUNTRY == "us":
                    continue

                if "anthro=True" in filepath:
                    continue

                pattern = r'=(\d+(\.\d+)?)'
                matches = re.findall(pattern, filepath)
                values = [match[0] for match in matches]
                qidx = int(values[0])

                save_dump_path = f"../dumps_wvs_2/q={qidx}_country={COUNTRY}_lang={LANG}_model={MODEL}_eval={EVAL_METHOD}.csv"
        
                # if qidx == 77 and COUNTRY == "egypt":
                #     print(f"> Skipping Q77")
                #     continue

                if qidx not in wvs_question_map:
                    print(f"> Skipping Q{values[0]}")
                    continue

                if not SCALE_QS and qidx in wvs_scale_questions:
                    print(f"> Skipping Q{values[0]} (Scale Q)")
                    continue
            
                if SCALE_QS and qidx not in wvs_scale_questions:
                    print(f"> Skipping Q{values[0]} (Not Scale Q)")
                    continue

                if str(qidx) not in wvs_response_map or f"Q{qidx}" not in wvs_questions:
                    print(f"> Skipping Q{values[0]}")
                    continue

                question_options = list(map(str.lower,wvs_questions[f"Q{qidx}"]["options"]))
                model_data = read_json(filepath)
                
                if len(model_data) != 4800 and COUNTRY == "egypt" and not ("AceGPT" in MODEL or 'Llama-2' in MODEL):
                    filepath = filepath.replace("_v3", "_v1")
                    filepath = filepath.replace("_maxt=16", "_maxt=8")
                    model_data = read_json(filepath)
                
                # try:
                #     if "AceGPT" in MODEL or 'Llama-2' in MODEL:
                #         assert len(model_data) >= 1212
                #     elif COUNTRY == "egypt":
                #         assert len(model_data) == 4800
                #     elif COUNTRY == "us":
                #         assert len(model_data) == 1200
                # except:
                #     print(len(model_data))
                
                model_df, invalid_count = convert_to_dataframe(model_data, question_options, demographic_map, eval_method=EVAL_METHOD, language=LANG)
                
                # if invalid_count > 0:
                #     print(f"q={qidx}_country={COUNTRY}_lang={LANG}_model={MODEL}")
                #     print(f"Invalid: {invalid_count}/{len(model_data)}")
                #     print()

                all_invalids += invalid_count 
                all_num_responses += len(model_data)

                model_df.to_csv(save_dump_path)
            
            print(f"{MODEL} | {LANG}: {all_invalids}/{all_num_responses}")
    #             if invalid_count >= 10:
    #                 # invalid_question_indices += [qidx]
    #                 print(invalid_count, os.path.basename(save_dump_path))

    # print(invalid_question_indices)