import re
import json
from pathlib import Path
from collections import defaultdict

output_dir = Path('data/cuad/reformatted')
output_dir.mkdir(parents=True, exist_ok=True)


def clean_context(context):
    # remove multiple spaces
    space_pattern = re.compile(r'[ \t]+')
    context = space_pattern.sub(' ', context)
    return context

def clean_answers(answers):
    answers = [a['text'] for a in answers]
    answers = [a.strip() for a in answers]
    answers = [a for a in answers if a]
    
    space_pattern = re.compile(r'[ \t]+')
    answers = [space_pattern.sub(' ', a) for a in answers]
    return answers


def reformat_data(dataset):
    new_dataset = []
    for x in dataset:
        for i in range(len(x['paragraphs'])):
            context = clean_context(x['paragraphs'][i]['context'])
        qas = x['paragraphs'][i]['qas']
        questions_answers_dict = defaultdict(list)
        for q in qas:
            questions_answers_dict[q['question']].extend(clean_answers(q['answers']))

        for q, a in questions_answers_dict.items():
            new_dataset.append({
                'context': context,
                'question': q,
                'answer': a,
            })
    return new_dataset


# run it for the whole test dataset
with open('data/cuad/test.json', 'r') as f:
    test_data = json.load(f)

test_data = test_data['data']
new_test_dataset = reformat_data(test_data)

# run it for the whole train dataset
with open('data/cuad/train_separate_questions.json', 'r') as f:
    train_data = json.load(f)

train_data = train_data['data']
new_train_dataset = reformat_data(train_data)

# convert to datasets
from datasets import Dataset
test_dataset = Dataset.from_list(new_test_dataset)
train_dataset = Dataset.from_list(new_train_dataset)

# save
test_dataset.to_json(str(output_dir / 'test.json'))
train_dataset.to_json(str(output_dir / 'train.json'))