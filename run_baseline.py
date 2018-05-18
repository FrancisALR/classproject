import sys

from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score


def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions = baseline.test(data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    report_score(gold_labels, predictions)

if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')
