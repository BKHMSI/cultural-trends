import os
import argparse
import pandas as pd
from tqdm import tqdm
from wvs_dataset import WVSDataset
from utils import read_json, write_json, retry_request, read_file

API_KEY = os.getenv("OPENAI_API_SOCIAL")
API_ORG = os.getenv("OPENAI_ORG_SOCIAL")

# url = "https://api.openai.com/v1/completions"
url = "https://api.openai.com/v1/chat/completions"
api_headers = {'Content-type': 'application/json', 'Accept': 'application/json',
               'Authorization': f'Bearer {API_KEY}', 'OpenAI-Organization': API_ORG}

def query_question(
    qid: int,
    *,
    model_name: str = 'gpt-3.5-turbo-0613',
    version: int = 1,
    lang: str = 'en',
    country: str = 'egypt',
    max_tokens: int = 3,
    temperature: float = 0.7,
    n_gen: int = 5,
    batch_size: int = 4,
    generator = None,
    tokenizer = None,
    no_persona: bool = False,
    subset = None,
    use_anthro: bool = False,
):
    qid = int(qid)
    filepath = f"../dataset/wvs_template.{lang}.yml"
    dataset = WVSDataset(filepath, 
        language=lang, 
        country=country, 
        api=True, 
        model_name=model_name,
        use_anthro_prompt=use_anthro
    )
    # selected_questions = read_file("../dataset/selected_questions.csv")[0].split(",")
    # selected_questions = list(map(str.strip, selected_questions))
    # selected_questions = [int(qnum[1:]) for qnum in selected_questions]

    selected_questions = dataset.question_ids

    if subset is not None:
        subset = int(subset)
        split_1 = int(len(selected_questions)*0.25)
        split_2 = int(len(selected_questions)*0.5)
        split_3 = int(len(selected_questions)*0.75)
        if subset == 1:
            selected_questions = selected_questions[:split_1]
        elif subset == 2:
            selected_questions = selected_questions[split_1:split_2]
        elif subset == 3:
            selected_questions = selected_questions[split_2:split_3]
        elif subset == 4:
            selected_questions = selected_questions[split_3:]
                
    if qid > 0: selected_questions = [f"Q{qid}"]
    
    print("="*50)
    print(f"Quering {len(selected_questions)} Questions")
    print("="*50)

    for qid in selected_questions:
        qid = int(qid[1:])
        if qid == 234: continue 
        if f"Q{qid}" not in dataset.question_ids:
            print(f"> Can't Find Q{qid}")
            continue

        dataset.set_question(index=qid)
        max_gens = len(dataset)

        filesuffix = f"q={str(qid).zfill(2)}_lang={lang}_country={country}_temp={temperature}_maxt={max_tokens}_n={n_gen}_persona={int(not no_persona)}_anthro={use_anthro}_v{version}"

        dirpath = f"../outputs_wvs_2/{model_name}"
        savedir = f"../results_wvs_2/{model_name}/{lang}"
        if not os.path.isdir(dirpath): os.makedirs(dirpath)
        if not os.path.isdir(savedir): os.makedirs(savedir)

        usage_path = os.path.join(dirpath, f"usage_{filesuffix}.json")
        preds_path = os.path.join(savedir, f"preds_{filesuffix}.json")
    
        completions, usage = [], []
        if os.path.exists(preds_path):
            completions = read_json(preds_path)
            # usage = read_json(usage_path)

        # assert len(completions) == len(usage)
        start_idx = len(completions)
        if start_idx >= max_gens:
            print(f"> Skipping Q{qid}")

        print(filesuffix)
        print(f"> Q{qid} | Prompting {model_name} | P{start_idx} to P{max_gens}")

        for index in tqdm(range(start_idx, max_gens)):
            payload_data, persona, q_info = dataset[index]
            payload = {"messages": [payload_data], "max_tokens": max_tokens,
                    "temperature": temperature, "model": model_name, "n": n_gen}
            
            response = retry_request(url, payload, api_headers)

            if "choices" in response:
                answers = [choice["message"]["content"].strip() for choice in response["choices"]]
                completions += [{
                    "persona": persona,
                    "question": q_info,
                    "response": answers,
                }]
                usage += [response["usage"]]
            else:
                print("> Error!")
                completions += [{"Error": response}]
                usage += [{"Error": 0}]
                
            write_json(preds_path, completions)
            write_json(usage_path, usage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # python wvs_query_model.py --model gpt-3.5-turbo-instruct --subset 1 --max-tokens 5 --n-gen 5 --lang en

    parser.add_argument('--qid', default=-1, type=int, help='question index')
    parser.add_argument('--model', default="gpt-3.5-turbo-0613", help='model to use')
    parser.add_argument('--version', default=7, type=int, help='dataset version number')
    parser.add_argument('--lang', default="en", help='language')
    parser.add_argument('--country', default="egypt", choices=["egypt", "us"], help='country')
    parser.add_argument('--max-tokens', default=10, type=int, help='maximum number of output tokens')
    parser.add_argument('--temperature', default=0.7, type=float, help='temperature')
    parser.add_argument('--n-gen', default=5, type=int, help='number of generations')
    parser.add_argument('--no-persona', default=False, action='store_true', help='whether to use persona')
    parser.add_argument('--use-anthro', default=False, action='store_true', help='whether to use anthro prompting')
    parser.add_argument('--subset', default=None, type=int, help='choose quartile', choices=[1,2,3,4])

    args = parser.parse_args()

    query_question(
        qid=args.qid,
        model_name=args.model,
        version=args.version,
        lang=args.lang,
        country=args.country,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        n_gen=int(args.n_gen),
        no_persona=args.no_persona,
        subset=args.subset,
        use_anthro=args.use_anthro
    )