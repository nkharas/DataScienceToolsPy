import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataVisualizations:
    def __init__(self, df, output_directory="Output", subsample=100, random_state=None):
        """
        :param df: Data frame to visualize
        :param remove_outliers: Remove outliers to make the graph more viewable. Default is False
                                If True, outliers selected based on percentile
        :param output_directory: Directory to save all visualization outputs.
                                 Default is to create an "Output" directory in the same path as the script.
        :param subsample: What percent of the data to randomly select as a subsample for visualizations? Default 100%
                          Useful when working with large volumes of data, enabling quicker overview and reduced cluttering from a lot of data points.
        :param random_state: Seed for the random number generator (if int), or numpy RandomState object when subsetting data.
        """
        if subsample < 100:
            self.df = df.sample(frac=subsample/100, random_state=random_state)
        else:
            self.df = df

        self.output_directory = output_directory

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    
    def visualize_numeric(self, column, nbins=100, outlier_cutoff=0):
        """
        Create histogram that examines the distribution of raw data across a continuous variable
        :param column: Name of the column in the dataframe
        :param nbins: Number of bins to create in the histogram. Default 100.
                      Decide this based on the range of values in column and the number of rows.
                      If possible number of bins is lower than nbins, use the former instead
        :param outlier_cutoff: Top and bottom percentiles to remove from graph. Default is 0 to not remove outliers.
                               Range should be 0 to 1. For example, to remove top and bottom 5% outliers, outlier_cutoff=0.05
        """
        x = self.df[column]

        # Calculate the bin width based on the range of raw data values and the number of bins to create
        bin_width = int((np.max(x) - np.min(x)) / nbins)
        
        # If possible number of bins is lower than nbins, use the former instead
        if bin_width == 0:
            bin_width = 1
        
        bins = range(int(np.min(x) - 1), int(np.max(x)+ bin_width), bin_width)
        
        plt.hist(x, bins)

        # Remove outliers from graph
        if outlier_cutoff > 0:
            left = np.min(x[x > np.percentile(x, outlier_cutoff*100)])
            right = np.max(x[x < np.percentile(x, (1-outlier_cutoff)*100)])
            plt.xlim(left, right)

        # Set title and label exes
        plt.title("Distribution of data across " + column)
        plt.xlabel(column)
        plt.ylabel("Frequency")

        # Save and close
        plt.savefig(self.output_directory + os.path.sep + column + ".png")
        plt.close()


    def visualize_categorical(self, column, kind="bar"):
        """
        Create bar chart that examines the distribution of raw data across classes in a categorical variable
        :param column: Name of the column in the dataframe
        :param kind: Type of plot. Default is bar plot
        """
        ax = self.df[column].value_counts().plot(kind=kind, title="Distribution of data across " + column)
        
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        fig = ax.get_figure()

        fig.savefig(self.output_directory + os.path.sep + kind + "_" + column + ".png")
        plt.cla()
        plt.clf()
        plt.close(fig)


    def visualize_ynum_to_xnum(self, dependent_variable, independent_variable, outlier_cutoff=0):
        """
        Create scatter plot that examines the relatonship between two variables
        :param dependent_variable: The Y variable tha a model will predict
        :param independent_variable: The X variable that a model will train on
        :param outlier_cutoff: Top and bottom percentiles to remove from graph for both X and Y variables.
                               Default is 0 to not remove outliers.
                               Range should be 0 to 1. For example, to remove top and bottom 5% outliers, outlier_cutoff=0.05
        """
        y = self.df[dependent_variable]
        x = self.df[independent_variable]

        plt.scatter(x, y)

        # Remove outliers from graph
        if outlier_cutoff > 0:
            x_left = np.min(x[x > np.percentile(x, outlier_cutoff*100)])
            x_right = np.max(x[x < np.percentile(x, (1-outlier_cutoff)*100)])
            
            y_bottom = np.min(y[y > np.percentile(y, outlier_cutoff*100)])
            y_top = np.max(y[y < np.percentile(y, (1-outlier_cutoff)*100)])
            
            plt.xlim(x_left, x_right)
            plt.ylim(y_bottom, y_top)

        # Set title and label exes
        plt.title("Relationship between " + dependent_variable + " and " + independent_variable)
        plt.xlabel(independent_variable)
        plt.ylabel(dependent_variable)

        # Save and close
        plt.savefig(self.output_directory + os.path.sep + dependent_variable + "_" + independent_variable + ".png")
        plt.close()


    def visualize_ynum_to_xcat(self, dependent_variable, independent_variable, kind="bar"):
        """
        Create scatter plot that examines the relatonship between two variables
        :param dependent_variable: The Y variable tha a model will predict
        :param independent_variable: The X variable that a model will train on
        :param kind: Type of plot. Default is bar plot Supported type is bar and box plots
        """
        if kind == "box":
            ax = self.df.boxplot(column=dependent_variable, by=independent_variable)
        else:
            groupby_df = pd.DataFrame(self.df[[independent_variable, dependent_variable]].groupby([independent_variable]).sum())
            ax = groupby_df.plot(kind=kind)

        ax.set_title("Relationship between " + dependent_variable + " and " + independent_variable)
        ax.set_xlabel(independent_variable)
        ax.set_ylabel(dependent_variable)

        fig = ax.get_figure()

        fig.savefig(self.output_directory + os.path.sep + kind + "_" + dependent_variable + "_" + independent_variable + ".png")
        plt.cla()
        plt.clf()
        plt.close(fig)
