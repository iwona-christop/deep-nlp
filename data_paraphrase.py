import json, logging
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MAP_LABEL_TRANSLATION = {
    0: 'negative',
    1: 'neutral',
    2: 'positive',
}


def paraphrase(sentence):
    text = "paraphrase: " + sentence + " </s>"

    encoding = paraphrase_tokenizer.encode_plus(
        text, 
        max_length=512, 
        padding=True, 
        return_tensors="pt"
    )

    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    diverse_beam_outputs = paraphrase_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=512,
        early_stopping=True,
        num_beams=2,
        num_beam_groups=2,
        num_return_sequences=2,
        diversity_penalty = 0.70
    )

    return paraphrase_tokenizer.decode(
        diverse_beam_outputs[0], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loaded_data = load_dataset('arize-ai/beer_reviews_label_drift_neg')
    paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality").to(device)
    paraphrase_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality", device=device)
    training_data = loaded_data['training']

    paraphrased_train = []
    start = 2000 # next = 2500

    for i in tqdm(range(start, start+500)):
        data = training_data[i]
        data_line = {
            'label': int(data['label']),
            'text': paraphrase(data['text']).replace('paraphrasedoutput: ', ''),
        }
        paraphrased_train.append(data_line)

    save_path = Path('/content')
    # save_train_path = save_path / '000.json'
    num_path = str(start) + '.json'
    save_train_path = save_path / num_path

    print(f'Saving into: {save_train_path}')

    with open(save_train_path, 'wt') as f_write:
        for data_line in paraphrased_train:
            data_line_str = json.dumps(data_line)
            f_write.write(f'{data_line_str}\n')