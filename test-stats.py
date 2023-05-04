import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# load csv file into a pandas dataframe
df = pd.read_csv('codedqr-test.csv')

df['pbmgs*10^-2'] = [d * 1e-2 for d in df['pbmgs*10^-2']]

# create a PDF file to save the plots
pdf_file = 'timing-plots.pdf'
pdf_pages = PdfPages(pdf_file)

# Get a list of the configuration variables
config_cols = ['p', 'n', 'f']

# Loop through the configuration variables and create a stacked bar chart for each one
for col in config_cols:
    # Group the data by all configuration variables except for the current one
    group_cols = [c for c in config_cols if c != col]
    grouped_means = df.groupby(group_cols)[df.columns[3:]].mean()

    # Loop through each unique value of the current configuration variable
    for val in df[col].unique():
        # Filter the original data for the current configuration value
        sub_df = df[df[col] == val]

        # Group the filtered data by the other configuration variables
        sub_grouped_means = sub_df.groupby(group_cols)[sub_df.columns[3:]].mean()
        sub_grouped_std = sub_df.groupby(group_cols)[sub_df.columns[3:]].std()

        # Create the stacked bar chart for the current configuration value
        ax = sub_grouped_means.plot(kind='bar', stacked=True, figsize=(10, 6), yerr=sub_grouped_std)
        
        # Set the title and axis labels for the plot
        ax.set_title(f'Timing Data vs. {", ".join([c for c in config_cols if c != col])}, {col}={val}')
        ax.set_xlabel(', '.join(group_cols))
        ax.set_ylabel('Time (s)')
        
        # Set the y-axis limits
        # ax.set_ylim(0, 5)
        
        # Set the y-axis to log scale
        # ax.set_yscale('log')
    
        # Save the figure to a PDF file
        pdf_pages.savefig()

# close the PDF file
pdf_pages.close()
