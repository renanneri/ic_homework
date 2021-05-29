import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import numpy as np


def plot_result_score_model(results,scoring,param):

    # For in scoring to make one graph to each scorer
    for scorer, color in zip(sorted(scoring), ['r','g','b','k']):
        plt.figure(figsize=(10, 10))
        plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
            fontsize=16)

        plt.xlabel(param)
        plt.ylabel("Score")

        ax = plt.gca()

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(results[f'param_{param}'].data, dtype=float)

    
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)


        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index] + 0.01, best_score + 0.010))

        plt.legend(loc="best")
        plt.grid(False)
        plt.show()