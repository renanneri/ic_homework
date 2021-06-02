import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import numpy as np
import pandas as pd


def plot_result_score_model(results,scoring,param):

    # For in scoring to make one graph to each scorer
    for scorer, color in zip(sorted(scoring), ['r','g','b','k']):
        plt.figure(figsize=(10, 10))
        plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
            fontsize=16)

        plt.xlabel(param)
        plt.ylabel("Score")

        ax = plt.gca()

        results_df = pd.DataFrame(results).groupby(['param_{}'.format(param)]).mean()
                
        max = pd.DataFrame(results).groupby(['param_{}'.format(param)]).max(numeric_only=True)
        min = pd.DataFrame(results).groupby(['param_{}'.format(param)]).min(numeric_only=True)
        # Get the regular numpy array from the MaskedArray
        X_axis = results_df.index

        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results_df['mean_%s_%s' % (sample, scorer)]
            min_sample = min['mean_%s_%s' % (sample, scorer)]
            max_sample = max['mean_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, min_sample,max_sample,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = results_df['rank_test_%s' % scorer].sort_values(ascending=False).index[0]
        worst_index = results_df['rank_test_%s' % scorer].sort_values(ascending=False).index[-1]
        best_score = results_df['mean_test_%s' % scorer][best_index]
        worst_score = results_df['mean_test_%s' % scorer][worst_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        
        ax.plot([best_index, ] * 2, [worst_score/1.1 , best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)


        # Annotate the best score for that scorer
        if(isinstance(best_index, float)):
            ax.annotate("%0.2f" % best_score,
                    (best_index + 0.01, best_score + 0.010))
        else:
            ax.annotate("%0.2f" % best_score, (best_index, best_score + 0.010))

        plt.legend(loc="best")
        plt.grid(False)
        plt.show()