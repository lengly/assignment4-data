import pandas as pd
import matplotlib.pyplot as plt

# read csv file
df = pd.read_csv('pretrain/mup_search_results.csv')

# create a figure with 3 subplots (3 rows, 1 column)
fig, axs = plt.subplots(3, 1, figsize=(7, 12))

# subplot 1: embeddings_scale vs valid_loss
axs[0].scatter(df['embeddings_scale'], df['valid_loss'], color='b')
axs[0].set_xlabel('embeddings_scale')
axs[0].set_ylabel('valid_loss')
axs[0].set_title('valid_loss vs embeddings_scale')
axs[0].grid(True)

# subplot 2: init_std vs valid_loss (log scale x-axis)
axs[1].scatter(df['init_std'], df['valid_loss'], color='g')
axs[1].set_xlabel('init_std')
axs[1].set_ylabel('valid_loss')
axs[1].set_title('valid_loss vs init_std')
axs[1].set_xscale('log')
axs[1].grid(True, which='both')

# subplot 3: lr vs valid_loss (log scale x-axis)
axs[2].scatter(df['lr'], df['valid_loss'], color='r')
axs[2].set_xlabel('lr')
axs[2].set_ylabel('valid_loss')
axs[2].set_title('valid_loss vs lr')
axs[2].set_xscale('log')
axs[2].grid(True, which='both')

plt.tight_layout()
plt.savefig('pretrain/mup_search_valid_loss_vs_all.png')
plt.close()

print('Done')
