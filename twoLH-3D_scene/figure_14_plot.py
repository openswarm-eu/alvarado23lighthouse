import pandas as pd
from functions.plotting import plot_acc_vs_mad


#############################################################################
###                                Options                                ###
#############################################################################

data_file = './figure_14_dataset.csv'

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df=pd.read_csv(data_file, index_col=0)

    # Plot Reconstruction Accuracy vs. Median Average Deviation
    plot_acc_vs_mad(df)


    
