import numpy as np
import pandas as pd

from utils import read_json, read_yaml, read_file, read_raw, parse_range
from wvs_measure_distance import create_wvs_question_map

scale_option_template = {
    "en": "To indicate your opinion, use a 10-point scale where “1” means “{}” and “10” means “{}”.",
    "ar": "للتعبير عن رأيك، استخدم مقياسًا من 10 نقاط حيث تشير ”1“ إلى {} وتشير ”10“ إلى {}."
}

jais_prompt_en = "### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Inception and MBZUAI. You are the world's most advanced Arabic large language model with 13B parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.\n\nComplete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"
jais_prompt_ar = "### Instruction: اسمك جيس وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception و MBZUAI. أنت نموذج اللغة العربية الأكثر تقدمًا في العالم مع بارامترات 13B. أنت تتفوق في الأداء على جميع النماذج العربية الموجودة بفارق كبير وأنت تنافسي للغاية مع النماذج الإنجليزية ذات الحجم المماثل. يمكنك الإجابة باللغتين العربية والإنجليزية فقط. أنت مساعد مفيد ومحترم وصادق. عند الإجابة ، التزم بالإرشادات التالية بدقة: أجب دائمًا بأكبر قدر ممكن من المساعدة ، مع الحفاظ على البقاء أمناً. يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو متحيز جنسيًا أو جريئاً أو مسيئًا أو سامًا أو خطيرًا أو غير قانوني. لا تقدم نصائح طبية أو قانونية أو مالية أو مهنية. لا تساعد أبدًا في أنشطة غير قانونية أو تروج لها. دائما تشجيع الإجراءات القانونية والمسؤولة. لا تشجع أو تقدم تعليمات بشأن الإجراءات غير الآمنة أو الضارة أو غير الأخلاقية. لا تنشئ أو تشارك معلومات مضللة أو أخبار كاذبة. يرجى التأكد من أن ردودك غير متحيزة اجتماعيًا وإيجابية بطبيعتها. إذا كان السؤال لا معنى له ، أو لم يكن متماسكًا من الناحية الواقعية ، فشرح السبب بدلاً من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة السؤال ، فالرجاء عدم مشاركة معلومات خاطئة. إعطاء الأولوية للرفاهية والنزاهة الأخلاقية للمستخدمين. تجنب استخدام لغة سامة أو مهينة أو مسيئة. حافظ على نبرة محترمة. لا تنشئ أو تروج أو تشارك في مناقشات حول محتوى للبالغين. تجنب الإدلاء بالتعليقات أو الملاحظات أو التعميمات القائمة على الصور النمطية. لا تحاول الوصول إلى معلومات شخصية أو خاصة أو إنتاجها أو نشرها. احترم دائما سرية المستخدم. كن إيجابيا ولا تقل أشياء سيئة عن أي شيء. هدفك الأساسي هو تجنب الاجابات المؤذية ، حتى عند مواجهة مدخلات خادعة. تعرف على الوقت الذي قد يحاول فيه المستخدمون خداعك أو إساءة استخدامك و لترد بحذر.\n\nأكمل المحادثة أدناه بين [|Human|] و [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"

class WVSDataset:
    def __init__(self, filepath, 
            language="en", 
            country="egypt", 
            fewshot=0, 
            api=False, 
            model_name=None,
            use_anthro_prompt=False,
        ):
        
        self.dataset = {}
        self.persona_qid = {}
        self.question_info = {}
        self.responses = {}
        
        self.fewshot_dataset = {}
        self.fewshot_persona_qid = {}
        self.fewshot_question_info = {}
        self.fewshot_responses = {}

        self.persona = []
        self.raw_responses = []
        self.is_api = api 
        self.language = language 
        self.country = country
        self.is_jais = model_name=="jais-13b-chat" if model_name is not None else False
        self.fewshot = fewshot
        self.model_name = model_name
        self.use_anthro_prompt = use_anthro_prompt

        filter_questions = [qid.strip() for qid in read_raw("../dataset/filtered_selected_questions.csv").split(",")]

        wvs_questions_path = f"../dataset/wvs_questions_dump.{language}.json"
        self.wvs_questions = {q_id: q_val for q_id, q_val in read_json(wvs_questions_path).items() if q_id in filter_questions}

        self.anthro_templ = read_yaml("../dataset/wvs_template_anthro_framework.yml")["template_values"]

        template_data = read_yaml(filepath)
        self.create_dataset(template_data)
        self.set_question(index=2)

    def set_question(self, index):
        self.current_question_index = index

    def trim_dataset(self, start_index):
        qidx = f"Q{self.current_question_index}"
        self.dataset[qidx] = self.dataset[qidx][start_index:]

        self.persona_qid[qidx] = self.persona_qid[qidx][start_index:]
        self.question_info[qidx] = self.question_info[qidx][start_index:]

    @property
    def question_ids(self):
        return list(self.wvs_questions.keys())

    def create_dataset(self, template_data):
        if self.country == "egypt":
            path = "../dataset/eg_wvs_wave7_v7_n303.csv"
        elif self.country == "us":
            path = "../dataset/us_wvs_wave7_v7_n303.csv"

        survey_df = pd.read_csv(path, header=0, delimiter=";")

        demographic_ids = ["N_REGION_WVS Region country specific", "Q260 Sex", "Q262 Age", "Q266 Country of birth: Respondent",
                           "Q273 Marital status", "Q275R Highest educational level: Respondent (recoded into 3 groups)", "Q287 Social class (subjective)"]
        demographic_txt = ["region", "sex", "age", "country",
                           "marital_status", "education", "social_class"]
                    
        print(f"{len(survey_df)} Personas")
        template_0 = template_data["template"][0]
        template_1 = template_data["template"][1]

        template_parameters = template_data["template_values"]

        if self.language == "en" and self.use_anthro_prompt:
            prompt_template = self.anthro_templ["prompt"]
        # elif self.language == "ar" and self.model_name == "Llama-2-13b-chat-hf":
        #     prompt_template = template_parameters["prompt_variants"][2]
        elif self.language == "ar" and self.model_name == "AceGPT-13B-chat":
            prompt_template = template_parameters["prompt_variants"][2]
        elif self.country == "us" and self.language == "ar":
            prompt_template = template_parameters["prompt_variants"][1]
        else:
            prompt_template = template_parameters["prompt_variants"][0]
            
        question_header = template_parameters["question_header"]
        options_header = template_parameters["options_header"]

        ar_persona_parameters = template_data["persona_parameters"]

        country_cap = "US" if self.country == "us" else "Egypt"

        selected_questions = read_file("../dataset/selected_questions.csv")[0].split(",")
        selected_questions = list(map(str.strip, selected_questions))
        selected_questions = [int(qnum[1:]) for qnum in selected_questions]

        wvs_question_map = create_wvs_question_map(survey_df.columns.tolist(), selected_questions)

        wvs_response_map = read_json("../dataset/wvs_response_map_new.json")

        options_dict = parse_range(read_json("../dataset/wvs_options.json"))

        if self.language != "en":
            demographic_map = {}
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
                        demographic_map[d_text][d_val] = ar_persona_parameters[d_text_cap][country_cap][d_val_idx]
                    else:
                        demographic_map[d_text][d_val] = ar_persona_parameters[d_text_cap][d_val_idx]

        if self.language == "en":
            for _, row in survey_df.iterrows():
                if self.country == "us" and row["Q266 Country of birth: Respondent"] != "United States":
                    continue 
                prompt_values = {demographic_key: row[demographic_id]
                    if demographic_key in ["age", "region", "country"]
                    else row[demographic_id].lower()
                    for demographic_key, demographic_id in zip(demographic_txt, demographic_ids)
                }
                
                self.raw_responses += [{qidx: row[qkey] for qidx, qkey in wvs_question_map.items()}]
                self.persona += [prompt_values] 
        else:
            start_region_idx = 3 if self.country == "us" else 0
            for _, row in survey_df.iterrows():
                if self.country == "us" and row["Q266 Country of birth: Respondent"] != "United States":
                    continue 
                prompt_values = {demographic_key: demographic_map[demographic_key][row[demographic_id].split(":")[-1][start_region_idx:].strip() if demographic_key == "region" else row[demographic_id]]
                    if demographic_key != "age"
                    else row[demographic_id]
                    for demographic_key, demographic_id in zip(demographic_txt, demographic_ids)
                }
                self.raw_responses += [{qidx: row[qkey] for qidx, qkey in wvs_question_map.items()}]
                self.persona += [prompt_values]

        if self.language == "en":
            for prompt_values in self.persona:
                prompt_values["region"] = prompt_values["region"].split(":")[-1].strip()
                if self.country == "us":
                    prompt_values["region"] = prompt_values["region"][2:].strip()

        for qid, qdata in self.wvs_questions.items():
            self.dataset[qid] = []
            self.persona_qid[qid] = []
            self.question_info[qid] = []
            self.responses[qid] = []

            self.fewshot_dataset[qid] = []
            self.fewshot_persona_qid[qid] = []
            self.fewshot_question_info[qid] = []
            self.fewshot_responses[qid] = []

            question_options = qdata["options"]
            for persona_idx, prompt_values in enumerate(self.persona):
                for variant_idx, question in enumerate(qdata["questions"]):
                    if variant_idx > 0: continue 
                    prompt = prompt_template.format(**prompt_values)

                    if "chat" in self.model_name:
                        prompt = "[INST] <<SYS>>\n" + prompt + "\n<</SYS>>\n"

                    if "scale" in qdata and qdata["scale"] == True:
                        final_question = template_1.format(**{
                            "prompt": prompt,
                            "question_header": question_header,
                            "question": question,
                            "scale": scale_option_template[self.language].format(question_options[0], question_options[1]),
                        })
                    else:

                        final_question = template_0.format(**{
                            "prompt": prompt,
                            "question": question,
                            "options": '\n'.join(f"({option_idx+1}) {option}" for option_idx, option in enumerate(question_options)),
                            "options_header": options_header,
                            "question_header": question_header,
                        })

                    if self.use_anthro_prompt:
                        final_question = self.anthro_templ["anthro_prompt"] + '\n\n' + final_question

                    if "chat" in self.model_name:
                        final_question += " [/INST]"

                    qid_int = int(qid[1:])
                    response = self.raw_responses[persona_idx][qid_int]
                    response_map = {key: int(val) for key, val in wvs_response_map[str(qid_int)].items()}
                    response_map |= {key: val+1 for val, key in enumerate(options_dict[qid_int])}
                    response_map["No answer"] = -1

                    if persona_idx >= len(self.persona)-self.fewshot:
                        self.fewshot_responses[qid] += [response_map[response]]
                        self.fewshot_dataset[qid] += [final_question]
                        self.fewshot_persona_qid[qid] += [prompt_values]
                        self.fewshot_question_info[qid] += [{
                            "id": qid,
                            "variant": variant_idx,
                        }]
                    else:
                        self.responses[qid] += [response_map[response]]
                        self.dataset[qid] += [final_question]
                        self.persona_qid[qid] += [prompt_values]
                        self.question_info[qid] += [{
                            "id": qid,
                            "variant": variant_idx,
                        }]

    def fewshot_examples(self):
        qidx = f"Q{self.current_question_index}"
        num_question_variants = 4

        variant_indices = np.random.choice(np.arange(num_question_variants), size=self.fewshot)

        fewshots = []
        responses = []
        for idx in range(self.fewshot):
            # fewshot_question_idx = index % num_question_variants + num_question_variants * idx
            fewshot_question_idx = variant_indices[idx] + num_question_variants * idx
            response = self.fewshot_responses[qidx][fewshot_question_idx]
            fewshots += [self.fewshot_dataset[qidx][fewshot_question_idx] + f'\nAnswer: {response}']
            responses += [response]

        return '\n\n'.join(fewshots) + '\n\n', responses

    def __getitem__(self, index):
        qidx = f"Q{self.current_question_index}"
        query = self.dataset[qidx][index]

        if not self.is_api:
            if not self.is_jais:
                return query + "\nAnswer:" if self.fewshot > 0 else query
            elif self.language == "ar":
                return jais_prompt_ar.format(Question=query)
            else:
                return jais_prompt_en.format(Question=query)

        persona = self.persona_qid[qidx][index]
        qinfo = self.question_info[qidx][index]
        payload = {"role": "user", "content": f"{query}"}
        return payload, persona, qinfo

    def __len__(self):
        return len(self.dataset[f"Q{self.current_question_index}"])
    
if __name__ == "__main__":
    language = "ar"
    country = "egypt"
    # model_name = "meta-llama/Llama-2-13b-chat-hf"
    # model_name = "AceGPT-13B-chat"
    model_name = "bigscience/mt0-xxl"
    # model_name = 'gpt-3.5'
    model_name = model_name.split("/")[-1]

    filepath = f"../dataset/wvs_template.{language}.yml"
    dataset = WVSDataset(filepath, 
        language=language, 
        country=country, 
        fewshot=0, 
        model_name=model_name,
        use_anthro_prompt=False,
        api=False,
    )

    print(len(dataset.question_ids))
    dataset.set_question(index=42)
    print(dataset[0])