import sys

from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score


def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])
    # for sent in data.trainset:
    #     print(sent['target_word'])
    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions = baseline.test(data.devset)
    # print(predictions)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    report_score(gold_labels, predictions)


# def execute_bayesian(language):
#     data = Dataset(language)
#
#     print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))
#
#     baseline = Baseline(language)
#
#     baseline.train(data.trainset)
#
#     predictions = baseline.test(data.devset)
#
#     gold_labels = [sent['gold_label'] for sent in data.devset]
#
#     report_score(gold_labels, predictions)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')