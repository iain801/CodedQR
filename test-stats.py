import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# load csv file into a pandas dataframe
df = pd.read_csv('codedqr-test.csv')

df['total'] = df.iloc[:, 3:].sum(axis=1)

df = df[['n', 'p', 'f', 'recovery','final solve','post-ortho','cs construct','pbmgs','total']]

seconds = int(df['total'].sum())
mins = seconds // 60
hours = mins // 60
days = hours // 24
seconds %= 60
mins %= 60
hours %= 24

print(f'total time: {days}-{hours}:{mins}:{seconds}')


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
df['decode'] = df['recovery'] / df['tqr']
df['comp'] = df['tcomp'] / df['tqr']

alpha = 5e0
gamma = 1e-2
df['f/p'] = gamma * df['f'] / df['p']
df['f/n'] = alpha * df['f'] / df['n']

# create a PDF file to save the plots
pdf_file = 'timing-plots.pdf'
pdf_pages = PdfPages(pdf_file)

# Get a list of the configuration variables
config_cols = ['n', 'p', 'f']

df_means = df.groupby(config_cols)[df.columns[3:]].mean()
df_std = df.groupby(config_cols)[df.columns[3:]].std()

# Create the stacked bar chart for the current configuration value
total_time = df_means.plot(y='total', kind='bar', stacked=False, figsize=(10, 8), yerr=df_std, color=['#ef6f6c','#8963ba','#90c290'])

# Set the title and axis labels for the plot
total_time.set_title(f'Total Execution Time')
total_time.set_xlabel(', '.join(config_cols))
total_time.set_ylabel('Time (s)')

# Save the figure to a PDF file
pdf_pages.savefig()

# Create the stacked bar chart for the current configuration value
overhead = df_means.plot(y=['comp', 'encode', 'post'], kind='bar', stacked=True, figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

# Set the title and axis labels for the plot
overhead.set_title(f'Overhead Breakdown')
overhead.set_xlabel(', '.join(config_cols))
overhead.set_ylabel('Proportion of  Tqr')

pdf_pages.savefig()

df_means = df.groupby(config_cols[1:])[df.columns[3:]].mean()
df_std = df.groupby(config_cols[1:])[df.columns[3:]].std()

# Create the stacked bar chart for the current configuration value
total_time = df_means.plot(y='total', kind='bar', stacked=False, figsize=(10, 8), color=['#ef6f6c','#8963ba','#90c290'])

# Set the title and axis labels for the plot
total_time.set_title(f'Total Execution Time')
total_time.set_xlabel(', '.join(config_cols))
total_time.set_ylabel('Time (s)')

# Save the figure to a PDF file
pdf_pages.savefig()

# Create the stacked bar chart for the current configuration value
overhead2 = df_means.plot(y=['comp', 'encode', 'post'], kind='bar', stacked=True, figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

# Set the title and axis labels for the plot
overhead2.set_title(f'Overhead Breakdown p, f')
overhead2.set_xlabel(', '.join(config_cols))
overhead2.set_ylabel('Proportion of  Tqr')

# Save the figure to a PDF file
pdf_pages.savefig()

# Create the stacked bar chart for the current configuration value
overhead3 = df_means.plot(y=['encode', 'post'], kind='bar', stacked=True, figsize=(10, 8), color=['#fff7ae','#916c80'])

# Set the title and axis labels for the plot
overhead3.set_title(f'Coding Breakdown')
overhead3.set_xlabel(', '.join(config_cols))
overhead3.set_ylabel('Proportion of  Tqr')

# Save the figure to a PDF file
pdf_pages.savefig()

# # Create the stacked bar chart for the current configuration value
# overhead3 = df_means.plot(y=['cs construct'], kind='bar', stacked=True, figsize=(10, 8), color=['#916c80'])

# # Set the title and axis labels for the plot
# overhead3.set_title(f'Absolute Encoding')
# overhead3.set_xlabel(', '.join(config_cols))
# overhead3.set_ylabel('Time (s)')

# # Save the figure to a PDF file
# pdf_pages.savefig()

# Create the stacked bar chart for the current configuration value
estimate = df_means.plot(y=['f/p'], kind='bar', stacked=False, figsize=(10, 8), color=['#916c80'])

# Set the title and axis labels for the plot
estimate.set_title(f'Coding Estimates ({gamma} * f/p)')
estimate.set_xlabel(', '.join(config_cols))
estimate.set_ylabel('Proportion of  Tqr')

pdf_pages.savefig()

df_means = df.groupby(['n','f'])[df.columns[3:]].mean()
df_std = df.groupby(['n','f'])[df.columns[3:]].std()

# Create the stacked bar chart for the current configuration value
overhead2 = df_means.plot(y=['encode', 'decode'], kind='bar', stacked=True, figsize=(10, 8), color=['#fff7ae','#916c80','#ffc6d9'])

# Set the title and axis labels for the plot
overhead2.set_title(f'Overhead Breakdown n, f')
overhead2.set_xlabel(', '.join(config_cols))
overhead2.set_ylabel('Proportion of  Tqr')

# Save the figure to a PDF file
pdf_pages.savefig()

# Create the stacked bar chart for the current configuration value
estimate2 = df_means.plot(y=['f/n'], kind='bar', stacked=False, figsize=(10, 8), color=['#ffc6d9'])

# Set the title and axis labels for the plot
estimate2.set_title(f'Coding Estimates ({alpha} * f/n)')
estimate2.set_xlabel(', '.join(config_cols))
estimate2.set_ylabel('Proportion of  Tqr')

pdf_pages.savefig()

# close the PDF file
pdf_pages.close()
