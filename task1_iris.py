# ============================================================
# Task 1: Exploring and Visualizing the Iris Dataset
# DevelopersHub Corporation - Data Science Internship
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ── 1. Load Dataset ──────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target_names[iris.target]

# ── 2. Dataset Structure ─────────────────────────────────────
print("=" * 50)
print("DATASET SHAPE:", df.shape)
print("=" * 50)
print("\nCOLUMNS:", df.columns.tolist())
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nBASIC STATISTICS:")
print(df.describe())
print("\nMISSING VALUES:")
print(df.isnull().sum())
print("\nSPECIES COUNT:")
print(df['species'].value_counts())

# ── 3. Visualizations ────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Task 1: Iris Dataset - Exploration & Visualization",
             fontsize=16, fontweight='bold', y=0.98)

# --- Scatter Plot: Sepal Length vs Sepal Width ---
ax1 = fig.add_subplot(2, 3, 1)
for species, group in df.groupby('species'):
    ax1.scatter(group['sepal_length'], group['sepal_width'],
                label=species, alpha=0.8, edgecolors='k', linewidths=0.4)
ax1.set_xlabel('Sepal Length (cm)')
ax1.set_ylabel('Sepal Width (cm)')
ax1.set_title('Scatter: Sepal Length vs Width')
ax1.legend()

# --- Scatter Plot: Petal Length vs Petal Width ---
ax2 = fig.add_subplot(2, 3, 2)
for species, group in df.groupby('species'):
    ax2.scatter(group['petal_length'], group['petal_width'],
                label=species, alpha=0.8, edgecolors='k', linewidths=0.4)
ax2.set_xlabel('Petal Length (cm)')
ax2.set_ylabel('Petal Width (cm)')
ax2.set_title('Scatter: Petal Length vs Width')
ax2.legend()

# --- Histogram: All Features ---
ax3 = fig.add_subplot(2, 3, 3)
colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid']
for i, col in enumerate(df.columns[:-1]):
    ax3.hist(df[col], bins=20, alpha=0.6, label=col, color=colors[i])
ax3.set_xlabel('Value (cm)')
ax3.set_ylabel('Frequency')
ax3.set_title('Histogram: Feature Distributions')
ax3.legend(fontsize=8)

# --- Box Plot: All Features by Species ---
ax4 = fig.add_subplot(2, 3, 4)
df_melted = df.melt(id_vars='species', var_name='Feature', value_name='Value')
sns.boxplot(data=df_melted, x='Feature', y='Value', hue='species',
            ax=ax4, palette='Set2')
ax4.set_title('Box Plot: Features by Species')
ax4.set_xlabel('Feature')
ax4.set_ylabel('Value (cm)')
ax4.tick_params(axis='x', rotation=15)
ax4.legend(fontsize=7)

# --- Heatmap: Correlation ---
ax5 = fig.add_subplot(2, 3, 5)
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax5, linewidths=0.5)
ax5.set_title('Heatmap: Feature Correlation')

# --- Pair Plot (small version) using scatter matrix ---
ax6 = fig.add_subplot(2, 3, 6)
species_colors = {'setosa': 'blue', 'versicolor': 'orange', 'virginica': 'green'}
for species, group in df.groupby('species'):
    ax6.scatter(group['sepal_length'], group['petal_length'],
                label=species, alpha=0.7, color=species_colors[species])
ax6.set_xlabel('Sepal Length (cm)')
ax6.set_ylabel('Petal Length (cm)')
ax6.set_title('Scatter: Sepal vs Petal Length')
ax6.legend()

plt.tight_layout()
plt.savefig('/home/claude/task1_iris_visualizations.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Task 1] Visualizations saved!")
