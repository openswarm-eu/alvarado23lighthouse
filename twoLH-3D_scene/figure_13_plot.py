import pandas as pd
from functions.plotting import plot_acc_vs_npoints

#############################################################################
###                                Options                                ###
#############################################################################

data_file = 'figure_13_dataset.csv'

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df=pd.read_csv(data_file, index_col=0)

    # Remove coplanar outliers
    df = df.loc[ (df['Coplanar'] > 30) & (df['MAE'] < 200)]

    # Plot the data
    plot_acc_vs_npoints(df)


    
