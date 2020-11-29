import torch
import os

class CommonsenseQAProcessor:
    """Processor for the CommonsenseQA data set."""

    SPLITS = ['qtoken', 'rand']
    LABELS = ['A', 'B', 'C', 'D', 'E']

    TRAIN_FILE_NAME = 'train_{split}_split.jsonl'
    DEV_FILE_NAME = 'dev_{split}_split.jsonl'
    TEST_FILE_NAME = 'test_{split}_split_no_answers.jsonl'

    def __init__(self, split):
        if split not in self.SPLITS:
            raise ValueError(
                'split must be one of {", ".join(self.SPLITS)}.')

        self.split = split

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME.format(split=self.split)

        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, train_file_name)),
            'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME.format(split=self.split)

        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, dev_file_name)),
            'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME.format(split=self.split)

        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, test_file_name)),
            'test')

    def get_labels(self):
        return [0, 1, 2, 3, 4]

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            qid = line['id']
            question = tokenization.convert_to_unicode(line['question']['stem'])
            answers = np.array([
                tokenization.convert_to_unicode(choice['text'])
                for choice in sorted(
                    line['question']['choices'],
                    key=lambda c: c['label'])
            ])
            # the test set has no answer key so use 'A' as a dummy label
            label = self.LABELS.index(line.get('answerKey', 'A'))

            examples.append(
                InputExample(
                    qid=qid,
                    question=question,
                    answers=answers,
                    label=label))

        return examples
