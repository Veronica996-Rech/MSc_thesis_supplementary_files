# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 18:20:56 2025

@author: Veronica Wokibula
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns

# Load your PLIP interaction data plus HADDOCK cluster mean scores from CSV
# The CSV should have columns: PLA2, Mtb_protein, Hydrogen_bonds, Salt_bridges, Hydrophobic_contacts, Pi_stacking, Pi_cation, HADDOCK_score
data = pd.read_csv('pla2_mtb_complexes_plip_scores.csv', delimiter = ';')

# Define interaction weights (adjust as needed)
weights = {
    'Hydrogen_bonds': 1.5,
    'Salt_bridges': 2.0,
    'Pi_stacking': 1.2,
    'Pi_cation': 1.8,
    'Hydrophobic_contacts': 1.0
}

# Calculate composite interaction score weighted sum
def compute_composite_score(row):
    score = 0
    for interaction, weight in weights.items():
        score += row.get(interaction, 0) * weight
    return score

data['Composite_score'] = data.apply(compute_composite_score, axis=1)

# If HADDOCK_score is already a cluster mean, use distinct pairs without additional aggregation
agg_data = data.drop_duplicates(subset=['PLA2', 'Mtb_protein'])[
    ['PLA2', 'Mtb_protein', 'Composite_score', 'HADDOCK_score']].copy()

# Rank by composite score (higher is better interaction)
agg_data['Composite_rank'] = agg_data['Composite_score'].rank(ascending=False)

# Rank by HADDOCK score (lower is better)
#agg_data['HADDOCK_rank'] = agg_data['HADDOCK_score'].rank()

# Combined rank (average of composite and HADDOCK ranks)
#agg_data['Combined_rank'] = agg_data[['Composite_rank', 'HADDOCK_rank']].mean(axis=1)

# Filter top 5 complexes based on combined rank
top5 = agg_data.nsmallest(5, 'Composite_rank')

print("Top 5 PLA2-Mtb protein complexes by combined interaction ranking:")
print(top5)

# Statistical testing: Do composite scores differ significantly across Mtb proteins? (Non-parametric test)
groups = [group['Composite_score'].values for name, group in agg_data.groupby('Mtb_protein')]
kruskal_stat, p_value = kruskal(*groups)
print(f"Kruskal-Wallis test for Composite_score differences between Mtb proteins: H={kruskal_stat:.3f}, p={p_value:.3e}")

# Visualization: heatmap of composite scores for PLA2 vs Mtb proteins
pivot = agg_data.pivot(index='PLA2', columns='Mtb_protein', values='Composite_score')
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, cmap='Blues')
plt.title('Heatmap of Composite Interaction Scores (PLA2 vs Mtb proteins)')

# Save the figures as a PNG file
plt.savefig('composite_scores_heatmap.png', dpi=300, bbox_inches='tight')

