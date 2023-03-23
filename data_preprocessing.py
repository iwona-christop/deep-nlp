import json, logging
from datasets import load_dataset
from pathlib import Path


logger = logging.getLogger(__name__)

MAP_LABEL_TRANSLATION = {
    0: 'negative',
    1: 'neutral',
    2: 'positive',
}


if __name__ == '__main__':
    loaded_data = load_dataset('arize-ai/beer_reviews_label_drift_neg')
    logger.info(f'Loaded dataset arize-ai/beer_reviews_label_drift_neg: {loaded_data}')

    save_path = Path('data')
    save_train_path = save_path / 'train.json'
    save_valid_path = save_path / 'valid.json'
    save_test_path = save_path / 'test.json'
    if not save_path.exists():
        save_path.mkdir()

    data_train, data_valid, data_test = [], [], []
    for source_data, dataset, max_size in [
        (loaded_data['training'], data_train, None),
        (loaded_data['validation'], data_valid, None),
        (loaded_data['production'], data_test, None)
    ]:
        for i, data in enumerate(source_data):
            if max_size is not None and i >= max_size:
                break
            data_line = {
                'label': int(data['label']),
                'text': data['text'],
            }
            dataset.append(data_line)
    logger.info(f'Train: {len(data_train):6d}')

    for file_path, data_to_save in [
        (save_train_path, data_train),
        (save_valid_path, data_valid),
        (save_test_path, data_test)
    ]:
        print(f'Saving into: {file_path}')
        with open(file_path, 'wt') as f_write:
            for data_line in data_to_save:
                data_line_str = json.dumps(data_line)
                f_write.write(f'{data_line_str}\n')
