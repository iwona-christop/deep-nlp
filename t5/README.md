# t5-custom

This model is a modified version of [T5](https://huggingface.co/docs/transformers/model_doc/t5) fine-tuned on an [arize-ai/beer_reviews_label_drift_neg](https://huggingface.co/datasets/arize-ai/beer_reviews_label_drift_neg) dataset.

It achieves the following results on the evaluation set:
- Loss: $0.4507$
- Bleu: $19.0721$
- Accuracy: $1.0$
- Gen Len: $2.121$

## Model description

As mentioned, the solution is based on the T5 model ([`t5-base`](https://huggingface.co/t5-base)). To solve the classification problem, it was treated as a seq2seq problem, where the first sequence was the review text and the second was the label. In addition, the weights of the layers, whose indices were not equal to successive elements of the Fibonacci sequence, were frozen.


## Training and evaluation data

As mentioned, the model was fine-tuned on an [arize-ai/beer_reviews_label_drift_neg](https://huggingface.co/datasets/arize-ai/beer_reviews_label_drift_neg) dataset which consists of beer reviews written in English and labels for sentiment classification.

### Data Fields

- `label`: indicating if the review is positive (`2`), neutral (`1`) or negative (`0`),
- `text`: the review written in English.

### Data Splits

As the model achieved satisfying results after fine-tuning on the default dataset, further training was skipped.

|                    | train | validation | test  |
| ------------------ | ----- | ---------- | ----- |
| default            | 9000  | 1260       | 27742 |

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: `5e-05`
- train_batch_size: `8`
- eval_batch_size: `8`
- seed: `42`
- optimizer: `Adam` with `betas=(0.9,0.999)` and `epsilon=1e-08`
- lr_scheduler_type: `linear`
- num_epochs: `1.0`

### Training results

| Training Loss | Epoch | Step | Validation Loss | Bleu    | Accuracy | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:--------:|:-------:|
| 0.6066        | 0.22  | 250  | 0.5035          | 18.4904 | 1.0      | 2.121   |
| 0.4373        | 0.44  | 500  | 0.4795          | 18.6545 | 1.0      | 2.121   |
| 0.445         | 0.67  | 750  | 0.4560          | 18.96   | 1.0      | 2.121   |
| 0.4386        | 0.89  | 1000 | 0.4507          | 19.0721 | 1.0      | 2.121   |
