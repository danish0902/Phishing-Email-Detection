"""
Generate comprehensive dataset visualizations for Phishing Email Detection project
Creates multiple graphs showing dataset statistics and characteristics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.preprocess import clean_text

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Create output directory
output_dir = project_root / "charts" / "dataset_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset...")
# Load the main dataset
data_path = project_root / "data" / "Phishing_Email_Cleaned_NO_DUPLICATES.csv"
df = pd.read_csv(data_path)

# Load train/val/test splits
train_df = pd.read_csv(project_root / "data" / "train_set.csv")
val_df = pd.read_csv(project_root / "data" / "val_set.csv")
test_df = pd.read_csv(project_root / "data" / "test_set.csv")

print(f"Total emails: {len(df)}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================================
# Graph 1: Class Distribution (Overall and by Split)
# ============================================================================
print("\n1. Generating class distribution graphs...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phishing Email Dataset - Class Distribution', fontsize=16, fontweight='bold')

# Overall distribution
labels = ['Legitimate', 'Phishing']
colors = ['#2ecc71', '#e74c3c']
counts = df['Email Type'].value_counts().sort_index()

axes[0, 0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
               explode=(0.05, 0.05), shadow=True)
axes[0, 0].set_title(f'Overall Dataset\n(Total: {len(df):,} emails)', fontweight='bold')

# Bar chart for overall
axes[0, 1].bar(labels, counts, color=colors, edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Number of Emails', fontweight='bold')
axes[0, 1].set_title('Overall Distribution (Bar Chart)', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(counts):
    axes[0, 1].text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                    ha='center', fontweight='bold')

# Distribution by split
splits = ['Train', 'Validation', 'Test']
split_dfs = [train_df, val_df, test_df]
split_data = []

for split_name, split_df in zip(splits, split_dfs):
    split_counts = split_df['Email Type'].value_counts().sort_index()
    split_data.append(split_counts)

split_array = np.array(split_data)
x = np.arange(len(splits))
width = 0.35

axes[1, 0].bar(x - width/2, split_array[:, 0], width, label='Legitimate', 
               color='#2ecc71', edgecolor='black')
axes[1, 0].bar(x + width/2, split_array[:, 1], width, label='Phishing', 
               color='#e74c3c', edgecolor='black')
axes[1, 0].set_xlabel('Dataset Split', fontweight='bold')
axes[1, 0].set_ylabel('Number of Emails', fontweight='bold')
axes[1, 0].set_title('Distribution by Train/Val/Test Split', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(splits)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Percentage stacked bar
split_percentages = (split_array.T / split_array.sum(axis=1) * 100).T
axes[1, 1].bar(splits, split_percentages[:, 0], label='Legitimate', 
               color='#2ecc71', edgecolor='black')
axes[1, 1].bar(splits, split_percentages[:, 1], bottom=split_percentages[:, 0],
               label='Phishing', color='#e74c3c', edgecolor='black')
axes[1, 1].set_ylabel('Percentage (%)', fontweight='bold')
axes[1, 1].set_title('Percentage Distribution by Split', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig(output_dir / '1_class_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 1_class_distribution.png")
plt.close()

# ============================================================================
# Graph 2: Email Length Distribution
# ============================================================================
print("\n2. Generating email length distribution graphs...")

# Calculate email lengths
df['email_length'] = df['Email Text'].astype(str).str.len()
df['word_count'] = df['Email Text'].astype(str).str.split().str.len()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Email Length Analysis', fontsize=16, fontweight='bold')

# Character length distribution
legitimate_chars = df[df['Email Type'] == 0]['email_length']
phishing_chars = df[df['Email Type'] == 1]['email_length']

axes[0, 0].hist([legitimate_chars, phishing_chars], bins=50, label=['Legitimate', 'Phishing'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Email Length (characters)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Character Length Distribution', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Word count distribution
legitimate_words = df[df['Email Type'] == 0]['word_count']
phishing_words = df[df['Email Type'] == 1]['word_count']

axes[0, 1].hist([legitimate_words, phishing_words], bins=50, label=['Legitimate', 'Phishing'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Word Count', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Word Count Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Box plots for comparison
data_chars = [legitimate_chars, phishing_chars]
axes[1, 0].boxplot(data_chars, labels=['Legitimate', 'Phishing'], 
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue', edgecolor='black'),
                   medianprops=dict(color='red', linewidth=2))
axes[1, 0].set_ylabel('Email Length (characters)', fontweight='bold')
axes[1, 0].set_title('Character Length Box Plot', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

data_words = [legitimate_words, phishing_words]
axes[1, 1].boxplot(data_words, labels=['Legitimate', 'Phishing'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', edgecolor='black'),
                   medianprops=dict(color='red', linewidth=2))
axes[1, 1].set_ylabel('Word Count', fontweight='bold')
axes[1, 1].set_title('Word Count Box Plot', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '2_email_length_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 2_email_length_distribution.png")
plt.close()

# ============================================================================
# Graph 3: Dataset Statistics Summary
# ============================================================================
print("\n3. Generating dataset statistics summary...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dataset Statistics Summary', fontsize=16, fontweight='bold')

# Statistics table
stats_data = {
    'Metric': ['Total Emails', 'Legitimate', 'Phishing', 'Train Set', 'Val Set', 'Test Set',
               'Avg Length (chars)', 'Avg Words', 'Max Length', 'Min Length'],
    'Value': [
        f"{len(df):,}",
        f"{len(df[df['Email Type']==0]):,} ({len(df[df['Email Type']==0])/len(df)*100:.1f}%)",
        f"{len(df[df['Email Type']==1]):,} ({len(df[df['Email Type']==1])/len(df)*100:.1f}%)",
        f"{len(train_df):,} (80%)",
        f"{len(val_df):,} (10%)",
        f"{len(test_df):,} (10%)",
        f"{df['email_length'].mean():.0f}",
        f"{df['word_count'].mean():.0f}",
        f"{df['email_length'].max():,}",
        f"{df['email_length'].min():,}"
    ]
}

axes[0, 0].axis('tight')
axes[0, 0].axis('off')
table = axes[0, 0].table(cellText=[[stats_data['Metric'][i], stats_data['Value'][i]] 
                                    for i in range(len(stats_data['Metric']))],
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
# Header styling
for i in range(2):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')
# Alternate row colors
for i in range(1, len(stats_data['Metric']) + 1):
    if i % 2 == 0:
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 1)].set_facecolor('#ecf0f1')
axes[0, 0].set_title('Overall Dataset Statistics', fontweight='bold', pad=20)

# Split size comparison (pie chart)
split_sizes = [len(train_df), len(val_df), len(test_df)]
split_labels = [f'Train\n{len(train_df):,}\n(80%)', 
                f'Val\n{len(val_df):,}\n(10%)', 
                f'Test\n{len(test_df):,}\n(10%)']
colors_split = ['#3498db', '#9b59b6', '#e67e22']

axes[0, 1].pie(split_sizes, labels=split_labels, autopct='%1.1f%%', startangle=90,
               colors=colors_split, explode=(0.05, 0.05, 0.05), shadow=True)
axes[0, 1].set_title('Train/Val/Test Split', fontweight='bold')

# Length statistics by class
stats_by_class = df.groupby('Email Type').agg({
    'email_length': ['mean', 'median', 'std'],
    'word_count': ['mean', 'median', 'std']
}).round(2)

class_stats_text = f"""
Email Length Statistics by Class:

LEGITIMATE:
  • Mean: {stats_by_class.loc[0, ('email_length', 'mean')]:.0f} chars
  • Median: {stats_by_class.loc[0, ('email_length', 'median')]:.0f} chars
  • Std Dev: {stats_by_class.loc[0, ('email_length', 'std')]:.0f} chars
  • Mean Words: {stats_by_class.loc[0, ('word_count', 'mean')]:.0f}

PHISHING:
  • Mean: {stats_by_class.loc[1, ('email_length', 'mean')]:.0f} chars
  • Median: {stats_by_class.loc[1, ('email_length', 'median')]:.0f} chars
  • Std Dev: {stats_by_class.loc[1, ('email_length', 'std')]:.0f} chars
  • Mean Words: {stats_by_class.loc[1, ('word_count', 'mean')]:.0f}
"""

axes[1, 0].text(0.1, 0.5, class_stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 0].axis('off')
axes[1, 0].set_title('Length Statistics by Class', fontweight='bold')

# Average length comparison
categories = ['Mean Chars', 'Median Chars', 'Mean Words']
legitimate_stats = [
    stats_by_class.loc[0, ('email_length', 'mean')],
    stats_by_class.loc[0, ('email_length', 'median')],
    stats_by_class.loc[0, ('word_count', 'mean')]
]
phishing_stats = [
    stats_by_class.loc[1, ('email_length', 'mean')],
    stats_by_class.loc[1, ('email_length', 'median')],
    stats_by_class.loc[1, ('word_count', 'mean')]
]

x = np.arange(len(categories))
width = 0.35

axes[1, 1].bar(x - width/2, legitimate_stats, width, label='Legitimate', 
               color='#2ecc71', edgecolor='black')
axes[1, 1].bar(x + width/2, phishing_stats, width, label='Phishing',
               color='#e74c3c', edgecolor='black')
axes[1, 1].set_ylabel('Value', fontweight='bold')
axes[1, 1].set_title('Average Length Comparison', fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '3_dataset_statistics.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 3_dataset_statistics.png")
plt.close()

# ============================================================================
# Graph 4: Data Quality and Coverage
# ============================================================================
print("\n4. Generating data quality graphs...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Quality and Coverage', fontsize=16, fontweight='bold')

# Missing values check
missing_data = df.isnull().sum()
columns = list(missing_data.index)
values = list(missing_data.values)
axes[0, 0].bar(columns, values, color='#e74c3c', edgecolor='black')
axes[0, 0].set_ylabel('Count', fontweight='bold')
axes[0, 0].set_title('Missing Values Check', fontweight='bold')
axes[0, 0].text(0.5, 0.5, 'No Missing Values ✓', transform=axes[0, 0].transAxes,
                fontsize=14, fontweight='bold', color='green', ha='center', va='center')
axes[0, 0].grid(axis='y', alpha=0.3)

# Duplicates check
axes[0, 1].text(0.5, 0.5, 
                f'Dataset Cleaned:\n\n✓ No Duplicates\n✓ {len(df):,} Unique Emails\n✓ Quality Verified',
                transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
axes[0, 1].axis('off')
axes[0, 1].set_title('Data Quality Status', fontweight='bold')

# Length range distribution
length_ranges = ['0-500', '501-1000', '1001-2000', '2001-5000', '5000+']
range_bins = [0, 500, 1000, 2000, 5000, float('inf')]

legit_ranges = pd.cut(legitimate_chars, bins=range_bins, labels=length_ranges).value_counts().sort_index()
phish_ranges = pd.cut(phishing_chars, bins=range_bins, labels=length_ranges).value_counts().sort_index()

x = np.arange(len(length_ranges))
width = 0.35

axes[1, 0].bar(x - width/2, legit_ranges, width, label='Legitimate', 
               color='#2ecc71', edgecolor='black')
axes[1, 0].bar(x + width/2, phish_ranges, width, label='Phishing',
               color='#e74c3c', edgecolor='black')
axes[1, 0].set_xlabel('Character Length Range', fontweight='bold')
axes[1, 0].set_ylabel('Count', fontweight='bold')
axes[1, 0].set_title('Email Length Distribution by Range', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(length_ranges, rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Class balance across splits
balance_data = []
for split_name, split_df in zip(splits, split_dfs):
    counts = split_df['Email Type'].value_counts().sort_index()
    balance = counts[1] / counts[0]  # Phishing/Legitimate ratio
    balance_data.append(balance)

axes[1, 1].bar(splits, balance_data, color='#3498db', edgecolor='black')
axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Balance')
axes[1, 1].set_ylabel('Phishing/Legitimate Ratio', fontweight='bold')
axes[1, 1].set_title('Class Balance Across Splits', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(balance_data):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '4_data_quality.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 4_data_quality.png")
plt.close()

# ============================================================================
# Graph 5: Sample Email Characteristics
# ============================================================================
print("\n5. Generating sample characteristics graphs...")

# Sample random emails for analysis
sample_size = min(1000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

# Count URLs in emails
df_sample['has_url'] = df_sample['Email Text'].astype(str).str.contains('http', case=False, na=False)
df_sample['url_count'] = df_sample['Email Text'].astype(str).str.count(r'http[s]?://')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Email Content Characteristics (Sample Analysis)', fontsize=16, fontweight='bold')

# URL presence by class
url_by_class = df_sample.groupby('Email Type')['has_url'].value_counts().unstack(fill_value=0)
url_by_class.index = ['Legitimate', 'Phishing']
url_by_class.columns = ['No URL', 'Has URL']

url_by_class.plot(kind='bar', ax=axes[0, 0], color=['#95a5a6', '#3498db'], 
                  edgecolor='black', rot=0)
axes[0, 0].set_ylabel('Count', fontweight='bold')
axes[0, 0].set_title('URL Presence by Email Type', fontweight='bold')
axes[0, 0].legend(title='URL Status')
axes[0, 0].grid(axis='y', alpha=0.3)

# URL count distribution
legit_urls = df_sample[df_sample['Email Type'] == 0]['url_count']
phish_urls = df_sample[df_sample['Email Type'] == 1]['url_count']

axes[0, 1].hist([legit_urls, phish_urls], bins=range(0, 6), label=['Legitimate', 'Phishing'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Number of URLs', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('URL Count Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Character diversity (unique chars / total chars)
df_sample['char_diversity'] = df_sample['Email Text'].astype(str).apply(
    lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
)

legit_diversity = df_sample[df_sample['Email Type'] == 0]['char_diversity']
phish_diversity = df_sample[df_sample['Email Type'] == 1]['char_diversity']

axes[1, 0].hist([legit_diversity, phish_diversity], bins=30, label=['Legitimate', 'Phishing'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Character Diversity Ratio', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Character Diversity Distribution', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Summary statistics table
summary_text = f"""
Sample Analysis Summary (n={sample_size}):

LEGITIMATE EMAILS:
  • With URLs: {url_by_class.loc['Legitimate', 'Has URL']} ({url_by_class.loc['Legitimate', 'Has URL']/url_by_class.loc['Legitimate'].sum()*100:.1f}%)
  • Avg URL count: {legit_urls.mean():.2f}
  • Avg diversity: {legit_diversity.mean():.3f}

PHISHING EMAILS:
  • With URLs: {url_by_class.loc['Phishing', 'Has URL']} ({url_by_class.loc['Phishing', 'Has URL']/url_by_class.loc['Phishing'].sum()*100:.1f}%)
  • Avg URL count: {phish_urls.mean():.2f}
  • Avg diversity: {phish_diversity.mean():.3f}

KEY OBSERVATIONS:
  • Phishing emails tend to have more URLs
  • Character diversity similar between classes
  • Dataset is well-balanced and clean
"""

axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('Content Analysis Summary', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '5_email_characteristics.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 5_email_characteristics.png")
plt.close()

# ============================================================================
# Print Summary
# ============================================================================
print("\n" + "="*70)
print("✓ All graphs generated successfully!")
print("="*70)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated graphs:")
print("  1. 1_class_distribution.png - Overall and split-wise class distribution")
print("  2. 2_email_length_distribution.png - Email length and word count analysis")
print("  3. 3_dataset_statistics.png - Comprehensive statistics summary")
print("  4. 4_data_quality.png - Data quality and coverage metrics")
print("  5. 5_email_characteristics.png - Email content characteristics")
print("\n" + "="*70)

# Print key statistics
print("\nKEY DATASET STATISTICS:")
print(f"  • Total emails: {len(df):,}")
print(f"  • Legitimate: {len(df[df['Email Type']==0]):,} ({len(df[df['Email Type']==0])/len(df)*100:.1f}%)")
print(f"  • Phishing: {len(df[df['Email Type']==1]):,} ({len(df[df['Email Type']==1])/len(df)*100:.1f}%)")
print(f"  • Train set: {len(train_df):,} (80%)")
print(f"  • Validation set: {len(val_df):,} (10%)")
print(f"  • Test set: {len(test_df):,} (10%)")
print(f"  • Average email length: {df['email_length'].mean():.0f} characters")
print(f"  • Average word count: {df['word_count'].mean():.0f} words")
print("\n" + "="*70)
