# Investigating Cultural Alignment of Large Language Models

## Abstract
> The intricate relationship between language and culture has long been a subject of exploration within the realm of linguistic anthropology. Large Language Models (LLMs), promoted as repositories of collective human knowledge, raise a pivotal question: do these models genuinely encapsulate the diverse knowledge adopted by different cultures? Our study reveals that these models demonstrate greater cultural alignment along two dimensions—firstly, when prompted with the dominant language of a specific culture, and secondly, when pretrained with a  refined mixture of languages employed by that culture. We quantify cultural alignment by simulating sociological surveys, comparing model responses to those of actual survey participants as references. Specifically, we replicate a survey conducted in various regions of Egypt and the United States through prompting LLMs with different pretraining data mixtures in both Arabic and English with the personas of the real respondents and the survey questions. Further analysis reveals that misalignment becomes more pronounced for underrepresented personas and for culturally sensitive topics, such as those probing social values. Finally, we introduce Anthropological Prompting, a novel method leveraging anthropological reasoning to enhance cultural alignment. Our study emphasizes the necessity for a more balanced multilingual pretraining dataset to better represent the diversity of human experience and the plurality of different cultures with many implications on the topic of cross-lingual transfer.

## Repository Structure
```php
cultural-trends/
│
├── scripts/
│   ├── wvs_query_openai.py
│   ├── wvs_query_hf.py
│   ├── wvs_majority_vote.py
│   ├── wvs_compute_alignment.py
│   └── wvs_compute_alignment_anthro.py
│
├── dataset/
│   ├── wvs_template.en.yml
│   ├── wvs_questions_dump.en.json
│   └── ...
│
└── completions/
    ├── <model>/
    │   ├── <country>/
    │   │   ├── <lang>/
    │   │   │   └── <completion>.json
    │   │   └── ...
    │   └── ...
    └── ...
```

### Scripts

- `wvs_query_openai.py`: Script for querying OpenAI's GPT models with World Values Survey (WVS) questions.
- `wvs_query_hf.py`: Script for querying Hugging Face's transformers models with WVS questions.
- `wvs_majority_vote.py`: Script for computing majority vote on model responses.
- `wvs_compute_alignment.py`: Script for computing cultural alignment using standard prompting.
- `wvs_compute_alignment_anthro.py`: Script for computing cultural alignment using Anthropological Prompting.

### Completions
This directory structure serves as a template for storing model completions:

- `<model>`: Name of the LLM model (e.g., GPT-3.5).
- `<country>`: Name of the country (e.g., Egypt, US).
- `<lang>`: Language used for prompting (e.g., English, Arabic).
- `<completion>.json`: JSON file containing model completions for survey questions.

## Completion JSON Structure
```json
{
  "persona": {
    "region": "String",
    "sex": "String",
    "age": "Number",
    "country": "String",
    "marital_status": "String",
    "education": "String",
    "social_class": "String"
  },
  "question": {
    "id": "String",
    "variant": "Number"
  },
  "response": ["String"]
}
```


- `persona`: contains demographic information about the survey participant, such as region, sex, age, country, marital status, education, and social class. Each attribute is represented as a string or number.
- `question` contains information about the survey question, including its unique identifier ("id") and variant number ("variant").
- `response`: is an array of strings representing the model's responses to the survey question variants. Each element in the array corresponds to a different linguistic variation or paraphrase of the question.

## Citation
TODO
