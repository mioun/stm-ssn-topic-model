import numpy as np
from sklearn import svm

from model.evaluation.retrival_metrics import RetrivalMetrics

DATA_SET = 'bbc'

INPUT_PATH = f'model-input-data/{DATA_SET}'
CONFIG_PATH = 'network-configuration'
OUTPUT_PATH = f'model-output-data/{DATA_SET}-article-final'

models = ['lda']
total_purity = []
for N in [20]:
    f_score = {m: [] for m in models}
    purity = {m: [] for m in models}
    for i in range(5):
        for m in models:
            model_name = f'{m}_{N}_{i}_clustering_metrics'
            M: RetrivalMetrics = RetrivalMetrics.load(OUTPUT_PATH, model_name)
            f_score[m].append(M.classification_metrics.fscore)
            print(model_name, M.purity)
            purity[m].append(M.purity)
            lin_clf = svm.LinearSVC()
            lin_clf.fit(M.train_topic_probability, M.train_labels)
            pred = lin_clf.predict(M.test_topic_probability)
            acc = 0
            for p, l in zip(pred, M.test_labels):
                if p == l:
                    acc += 1
            print("ACC :", acc / len(M.test_labels))

    for m in models:
        print(
            f'f_score for {m} and {N} :{round(np.average(f_score[m]) * 100, 2)} ({round(np.std(f_score[m]) * 100, 2)})')

    for m in models:
        p = round(np.average(purity[m]) * 100, 2)
        print(f'purity for {m} and {N} :{p} ({round(np.std(purity[m]) * 100, 2)})')
        total_purity.append(p)

print(f'Total purity :{round(np.average(total_purity), 2)} ({round(np.std(total_purity), 2)})')
