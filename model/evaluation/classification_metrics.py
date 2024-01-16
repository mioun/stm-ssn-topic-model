import os.path

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import numpy as np


class ClassificationMetrics:

    def __init__(self, true_y: list, pred_y: list, categories: list):
        self.cat_mapping = categories
        self.pred_y = pred_y
        self.true_y = true_y
        self.precision = -1
        self.recall = -1
        self.fscore = -1
        self.fscore_per_class = {}
        self.precision_per_class = {}
        self.recall_per_class = {}
        self.accuracy = 0

       # self.calculate_results()

    def calculate_results(self):
        self.precision = precision_score(self.true_y, self.pred_y, average='macro')
        self.precision_per_class = {self.cat_mapping[idx]: prec for idx, prec in
                                    enumerate(precision_score(self.true_y, self.pred_y, average=None))}

        self.recall = recall_score(self.true_y, self.pred_y, average='macro')
        self.recall_per_class = {self.cat_mapping[idx]: prec for idx, prec in
                                 enumerate(recall_score(self.true_y, self.pred_y, average=None))}

        self.fscore = f1_score(self.true_y, self.pred_y, average='macro')
        self.fscore_per_class = {self.cat_mapping[idx]: prec for idx, prec in
                                 enumerate(f1_score(self.true_y, self.pred_y, average=None))}

        self.accuracy = accuracy_score(self.true_y, self.pred_y)

    def to_fancy_string(self):
        lines = []
        lines.append('#########' + ' Global results ' + '#########' + '\n')
        self.append_line(lines, ['accuracy', str(self.accuracy)])
        self.append_line(lines, ['precision', str(self.precision)])
        self.append_line(lines, ['recall', str(self.recall)])
        self.append_line(lines, ['fscore', str(self.fscore)])

        self.append_dict(lines, self.precision_per_class, ' Precision ')
        self.append_dict(lines, self.recall_per_class, ' Recall ')
        self.append_dict(lines, self.fscore_per_class, ' Fscore ')

        return lines

    def append_dict(self, lines, dictionary: dict, header: str):
        lines.append('#########' + header + '#########' + '\n')
        for key in dictionary:
            self.append_line(lines, [str(key), str(dictionary[key])])

    def append_line(self, lines: list, line: list):
        spliter = ' '
        lines.append(spliter.join(line) + '\n')

    def save_results(self, result_folder, suffix=''):
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        file_path = result_folder + "/" + suffix + "predictions.csv"
        arr2d = np.asarray([self.true_y, self.pred_y])
        np.savetxt(file_path, arr2d, delimiter=',')

        with open(result_folder + "/metryka" + suffix, 'w+') as file:
            predicates = self.to_fancy_string()
            for predicate in predicates:
                file.write(predicate)
            file.close()

