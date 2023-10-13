import sys
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

inpath = sys.argv[1]
outpath = sys.argv[2]

print(f'making plots from {inpath}')

# load csv file into a pandas dataframe
df = pd.read_csv(inpath)

df['total'] = df.iloc[:, 3:].sum(axis=1) - df['recovery']

df = df[['n', 'p', 'f', 'recovery','final solve','post-ortho','cs construct','pbmgs','total']]

seconds = int(df['total'].sum())
mins = seconds // 60
hours = mins // 60
days = hours // 24
seconds %= 60
mins %= 60
hours %= 24

print(f'total execution time: {days}-{hours}:{mins}:{seconds}')

# Add a new 'tqr' column initialized with the existing 'pbmgs' values
df['tqr'] = df['pbmgs']

# Create a dictionary mapping (n, p) tuples to corresponding tqr values where f is 0
f_zero_dict = df.loc[df['f'] == 0].set_index(['n', 'p'])['tqr'].to_dict()

# Update 'tqr' for all rows except where f is 0 using the values from the f_zero_dict
df.loc[df['f'] != 0, 'tqr'] = df.loc[df['f'] != 0].apply(
    lambda row: f_zero_dict.get((row['n'], row['p']), row['tqr']),
    axis=1
)

# add tcomp to df such that tcomp = pbmgs - tqr
df['tcomp'] = df['pbmgs'] - df['tqr']

df['encode'] = df['cs construct'] / df['tqr']
df['post'] = df['post-ortho'] / df['tqr']
df['decode'] = df['recovery'] / df['tqr'] / ( df['p'] + df['f'] )
df['comp'] = df['tcomp'] / df['tqr']

df['overhead'] = df['comp'] + df['encode'] + df['post']

alpha = 5e0
gamma = 1e-2
df['f/p'] = gamma * df['f'] / df['p']
df['f/n'] = alpha * df['f'] / df['n']

# create a PDF file to save the plots
pdf_pages = PdfPages(outpath)

# Get a list of the configuration variables
config_cols = ['n', 'p', 'f']

df_means = df.groupby(config_cols)[df.columns[3:]].mean()
df_std = df.groupby(config_cols)[df.columns[3:]].std()

# Create the bar chart for the current configuration value
total_time = df_means.plot(y='total', kind='bar',
                           figsize=(10, 8), color=['#ef6f6c','#8963ba','#90c290','#51AEF0'])

# Set the title and axis labels for the plot
total_time.set_title(f'Total Execution Time')
total_time.set_xlabel(', '.join(config_cols))
total_time.set_ylabel('Time (s)')

# Save the figure to a PDF file
pdf_pages.savefig()

# Create the stacked bar chart for the current configuration value
overhead = df_means.plot(y=['overhead'], kind='bar',
                         figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

# Set the title and axis labels for the plot
overhead.set_title(f'Overhead Breakdown')
overhead.set_xlabel(', '.join(config_cols))
overhead.set_ylabel('Proportion of  Tqr')

pdf_pages.savefig()

for n_val in df['n'].unique():
    
    n_rows = df_means.query(f"n == {n_val}")
    # n_rows.drop('n', axis=1, inplace=True)
    
    # Create the bar chart for the current configuration value
    total_time_n = n_rows.plot(y='total', kind='bar',
                               figsize=(10, 8), color=['#ef6f6c','#8963ba','#90c290','#51AEF0'])

    # Set the title and axis labels for the plot
    total_time_n.set_title(f'Total Execution Time n = {n_val}')
    total_time_n.set_xlabel(', '.join(config_cols[1:]))
    total_time_n.set_ylabel('Time (s)')

    # Save the figure to a PDF file
    pdf_pages.savefig()

    # Create the stacked bar chart for the current configuration value
    overhead_n = n_rows.plot(y=['overhead'], kind='bar',
                             figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

    # Set the title and axis labels for the plot
    overhead_n.set_title(f'Overhead Breakdown n={n_val}, p, f')
    overhead_n.set_xlabel(', '.join(config_cols[1:]))
    overhead_n.set_ylabel('Proportion of  Tqr')

    # Save the figure to a PDF file
    pdf_pages.savefig()
    
    # Create the stacked bar chart for the current configuration value
    coding_n = n_rows.plot(y=['encode', 'decode', 'post'], kind='bar',
                           figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

    # Set the title and axis labels for the plot
    coding_n.set_title(f'Coding Breakdown n={n_val}, p, f')
    coding_n.set_xlabel(', '.join(config_cols[1:]))
    coding_n.set_ylabel('Proportion of  Tqr')

    # Save the figure to a PDF file
    pdf_pages.savefig()

df_means = df.groupby(config_cols[1:])[df.columns[3:]].mean()
df_std = df.groupby(config_cols[1:])[df.columns[3:]].std()

overhead = df_means.plot(y=['overhead'], kind='bar', stacked=True, figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

# Set the title and axis labels for the plot
overhead.set_title(f'Overhead Average p, f')
overhead.set_xlabel(', '.join(config_cols[1:]))
overhead.set_ylabel('Proportion of  Tqr')

# Save the figure to a PDF file
pdf_pages.savefig()

# Create the stacked bar chart for the current configuration value
coding = df_means.plot(y=['encode', 'decode', 'post'], kind='bar', stacked=False, figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

# Set the title and axis labels for the plot
coding.set_title(f'Encode and Reocvery Breakdown p, f')
coding.set_xlabel(', '.join(config_cols[1:]))
coding.set_ylabel('Proportion of  Tqr')

# Save the figure to a PDF file
pdf_pages.savefig()

# close the PDF file
pdf_pages.close()

print(f'plots saved to {outpath}')
