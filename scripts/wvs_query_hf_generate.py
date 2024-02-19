import os
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from wvs_dataset import WVSDataset
from utils import read_json, write_json

from transformers.utils import logging
logging.set_verbosity(50)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate(model, tokenizer, fewshot_cache, prompts, device, n_steps=20):
    # generation cycle with 20 steps
    step = 0
    past_key_values = fewshot_cache
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    output = None
    while step < n_steps:
        attention_mask = input_ids.new_ones(input_ids.shape)
        
        if output is not None:
            past_key_values = output["past_key_values"]

        ids = model.prepare_inputs_for_generation(input_ids,
                                                past=past_key_values,
                                                attention_mask=attention_mask,
                                                use_cache=True)
                                    
        output = model(**ids)
        
        # next_token = random.choice(torch.topk(output.logits[:, -1, :], top_k, dim=-1).indices[0])
        next_token = output.logits[:, -1, :].argmax(dim=-1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        
        step += 1

    return input_ids


def query_hf(
    qid: str,
    *,
    model_name: str = 'bigscience/mt0-small',
    version: int = 1,
    lang: str = 'en',
    max_tokens: int = 8,
    temperature: float = 0.7,
    n_gen: int = 5,
    batch_size: int = 4,
    fewshot: int = 0,
    cuda: int = 0,
    greedy: bool = False,
    generator = None,
    tokenizer = None,
    no_persona = False,
    subset = None,
    country: str = "egypt",
):
    
    model_name_ = model_name.split("/")[-1]
    savedir = f"../results_wvs_2/{model_name_}/{lang}"
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    filepath = f"../dataset/wvs_template.{lang}.yml"

    dataset = WVSDataset(filepath, 
        language=lang, 
        country=country, 
        api=False, 
        model_name=model_name_,
        use_anthro_prompt=False
    )

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Language={lang} | Temperature={temperature} | Tokens={max_tokens} | N={n_gen} | Batch={batch_size} | Version={version}")
    print(f"> Device {device}")

    if qid <= 0:
        question_ids = dataset.question_ids
    else:
        question_ids = [f"Q{qid}"]
    
    print(f"> Running {len(question_ids)} Qs")
    model_path = model_name
    if "mt0" in model_name_:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float16).to(device)
    else:
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        if "AceGPT" in model_name:
            model_path = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/models/models--FreedomIntelligence--AceGPT-13B-chat/snapshots/ab87ccbc2c4a05969957755aaabc04400bb20052"
        elif "Llama" in model_name:
            model_path = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/models/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496" 
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float16).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if"Llama-2-13b-chat-hf" in model_name or "AceGPT-13B-chat" in model_name:
        print("> Changing padding side")
        tokenizer.padding_side = "left"

    if model_name == "gpt2" or "Sheared-LLaMA-1.3B" in model_name or "Llama-2-13b" in model_name or "AceGPT-13B-chat" in model_name:
        tokenizer.pad_token = tokenizer.eos_token

    for qid in question_ids:
        qid = int(qid[1:])
        dataset.set_question(index=qid)

        filesuffix = f"q={str(qid).zfill(2)}_lang={lang}_country={country}_temp={temperature}_maxt={max_tokens}_n={n_gen}_v{version}_fewshot={fewshot}"
        print(filesuffix)

        preds_path = os.path.join(savedir, f"preds_{filesuffix}.json")
        
        completions = []
        if os.path.exists(preds_path):
            completions = read_json(preds_path)

        if len(completions) >= len(dataset):
            print(f"Skipping Q{qid}")
            continue
        
        if len(completions) > 0:
            print(f"> Trimming Dataset from {len(completions)}")
            dataset.trim_dataset(len(completions))

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=False)

        if fewshot > 0:

            fewshot_examples, _ = dataset.fewshot_examples()
            fewshot_tokens = tokenizer(fewshot_examples, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                fewshot_cache = model(**fewshot_tokens, use_cache=True)["past_key_values"]

        index = 0
        print(f"> Prompting {model_name} with Q{qid}")
        for batch_idx, prompts in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            if fewshot == 0:                
                tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

                gen_outputs = model.generate(**tokens, 
                    temperature=temperature,
                    do_sample=(not greedy), 
                    num_return_sequences=n_gen, 
                    max_new_tokens=max_tokens,
                )
                decoded_output = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                for b_i in range(0, len(decoded_output), n_gen):
                    preds = decoded_output[b_i:b_i+n_gen]
                    preds = [pred.replace(prompts[b_i//n_gen], "") for pred in preds]
                    persona = dataset.persona_qid[f"Q{qid}"][index]
                    q_info = dataset.question_info[f"Q{qid}"][index]
                    index += 1
                    completions += [{
                        "persona": persona,
                        "question": q_info,
                        "response": preds,
                    }]

            else:
         
                # prompts_with_fewshot = [fewshot_examples + prompt for prompt in prompts]
                # tokens_with_fewshot = tokenizer(prompts_with_fewshot, padding=True, return_tensors="pt").to(device)

                # start_time = time.time()
                # gen_outputs_wo_cache = model.generate(**tokens_with_fewshot, 
                #     temperature=temperature,
                #     do_sample=(not greedy), 
                #     num_return_sequences=n_gen, 
                #     max_new_tokens=max_tokens,
                # )
            
                # decoded_output = tokenizer.batch_decode(gen_outputs_wo_cache, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
                # tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                # tokens["input_ids"] = tokens["input_ids"][:, 1:]
                # tokens["attention_mask"] = tokens["attention_mask"][:, 1:]

                # with torch.no_grad():
                #     prompt_cache = model(**tokens_with_fewshot, past_key_values=fewshot_cache, use_cache=True)["past_key_values"]

                # tokens_with_fewshot_concat = {
                #     "input_ids": torch.cat([fewshot_tokens["input_ids"].repeat(batch_size, 1), tokens["input_ids"][:, 1:]], dim=1),
                #     "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"][:, 1:]],dim=1),
                #     # "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"][:, 1:], torch.ones(batch_size, 1).to(device)],dim=1),
                # }

                # tokens_with_fewshot_concat = {
                #     "input_ids": torch.cat([fewshot_tokens["input_ids"].repeat(batch_size, 1), tokens["input_ids"]], dim=1),
                #     "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"]],dim=1),
                #     # "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"], torch.zeros(batch_size, 1).to(device)],dim=1),
                # }

                # num_layers = len(fewshot_cache)
                # all_cache = []
                # for layer_idx in range(num_layers):
                #     all_cache += [(
                #         torch.cat([fewshot_cache[layer_idx][0].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][0].repeat(n_gen,1,1,1)[:,:,:-1,:]], dim=2),
                #         torch.cat([fewshot_cache[layer_idx][1].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][1].repeat(n_gen,1,1,1)[:,:,:-1,:]], dim=2),
                #     )]
                    # all_cache += [(
                    #     torch.cat([fewshot_cache[layer_idx][0].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][0].repeat(n_gen,1,1,1)], dim=2),
                    #     torch.cat([fewshot_cache[layer_idx][1].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][1].repeat(n_gen,1,1,1)], dim=2),
                    # )]

                # print("Fewshot Tokens + Prompt Tokens: ", fewshot_tokens["input_ids"].size(1) + tokens["input_ids"].size(1))
                # print("[Fewshot, Prompt] Tokens: ", tokens_with_fewshot_concat["input_ids"].size(1))
                # # print("(Fewshot + Prompt) Tokens: ", tokens_with_fewshot["input_ids"].size(1))
                # print("Cache Concat: ", all_cache[0][0].size())
                # print("[Fewshot, Prompt] Attention Mask: ", tokens_with_fewshot_concat["attention_mask"].size(1))
                # breakpoint()
                # # del prompt_cache

                # tokens_with_fewshot["attention_mask"] = torch.cat([tokens_with_fewshot["attention_mask"], torch.ones(batch_size,1).to(device)], dim=1)

                # # start_time = time.time()
                # gen_outputs = model.generate(**tokens_with_fewshot_concat,
                #     # input_ids=tokens_with_fewshot_concat["input_ids"], 
                #     temperature=temperature,
                #     do_sample=(not greedy), 
                #     num_return_sequences=n_gen, 
                #     max_new_tokens=max_tokens,
                #     past_key_values=tuple(all_cache)
                # )

                gen_outputs = generate(model, tokenizer, fewshot_cache, prompts, device, n_steps=max_tokens)

                # gen_outputs = model.generate(**tokens_with_fewshot_concat, 
                #     temperature=temperature,
                #     do_sample=(not greedy), 
                #     num_return_sequences=n_gen, 
                #     max_new_tokens=max_tokens,
                #     past_key_values=tuple(all_cache)
                # )

                decoded_output = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                for b_i in range(0, len(decoded_output), n_gen):
                    preds = decoded_output[b_i:b_i+n_gen]
                    preds = [pred.replace(fewshot_examples, "").replace(prompts[b_i//n_gen], "") for pred in preds]
                    persona = dataset.persona_qid[f"Q{qid}"][index]
                    q_info = dataset.question_info[f"Q{qid}"][index]
                    index += 1
                    completions += [{
                        "persona": persona,
                        "question": q_info,
                        "response": preds,
                    }]

            write_json(preds_path, completions)

if __name__ == "__main__":

    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --lang en --fewshot 3 --cuda 0
    # python wvs_query_hf_generate.py --qid -1 --model princeton-nlp/Sheared-LLaMA-1.3B --max-tokens 10 --lang en --fewshot 3 --cuda 0
    # python wvs_query_hf_generate.py --qid -1 --model princeton-nlp/Sheared-LLaMA-1.3B --max-tokens 5 --lang ar --fewshot 3 --cuda 0 --n-gen 1
    # python wvs_query_hf_generate.py --qid -1 --model meta-llama/Llama-2-13b-chat-hf --max-tokens 16 --lang en --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4
    
    # python wvs_query_hf_generate.py --qid -1 --model meta-llama/Llama-2-13b-chat-hf --max-tokens 32 --lang ar --fewshot 0 --cuda 1 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --max-tokens 5 --lang ar --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --max-tokens 5 --lang en --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model bigscience/mt0-xxl --max-tokens 5 --lang en --fewshot 0 --cuda 1 --n-gen 5 --batch-size 4 --country us

    # cp -r /home/bkhmsi/.cache/huggingface/hub/models--FreedomIntelligence--AceGPT-13B-chat /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/models
    parser = argparse.ArgumentParser()

    parser.add_argument('--qid', required=True, type=int, help='question index')
    parser.add_argument('--model', default="bigscience/mt0-small", help='model to use')
    parser.add_argument('--version', default=1, help='dataset version number')
    parser.add_argument('--lang', default="en", help='language')
    parser.add_argument('--max-tokens', default=4, type=int, help='maximum number of output tokens')
    parser.add_argument('--temperature', default=0.7, type=float, help='temperature')
    parser.add_argument('--n-gen', default=5, type=int, help='number of generations')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('--fewshot', default=0, type=int, help='fewshot examples')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device number')
    parser.add_argument('--greedy', action="store_true", help='greedy decoding')
    parser.add_argument('--country', type=str, help='country')

    args = parser.parse_args()

    if args.greedy:
        args.n_gen = 1
        args.temperature = 1.0
    
    qid = int(args.qid)

    query_hf(
        qid=qid,
        model_name=args.model,
        version=args.version,
        lang=args.lang,
        max_tokens=args.max_tokens,
        temperature=float(args.temperature),
        n_gen=int(args.n_gen),
        batch_size=int(args.batch_size),
        fewshot=int(args.fewshot),
        cuda=args.cuda,
        greedy=args.greedy,
        country=args.country,
    )