"""
Distribution analysis without arbitrary thresholds
Following Kullback-Leibler divergence methodology
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

def compare_distributions(
    dist1: np.ndarray, 
    dist2: np.ndarray,
    labels: Tuple[str, str] = ('Distribution 1', 'Distribution 2')
) -> Dict:
    """
    Comprehensively compare two distributions.
    
    Uses multiple metrics following best practices from
    Ramdas et al. (2017) "On Wasserstein Two-Sample Testing" ArXiv.
    
    Returns:
        Dictionary with multiple comparison metrics
    """
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(dist1, dist2)
    
    # Wasserstein distance (Earth Mover's Distance)
    # Better than KS for measuring "how different" distributions are
    emd = wasserstein_distance(dist1, dist2)
    
    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(dist1) + np.var(dist2)) / 2)
    cohens_d = (np.mean(dist1) - np.mean(dist2)) / pooled_std
    
    # Mann-Whitney U test (non-parametric)
    mw_stat, mw_pval = stats.mannwhitneyu(dist1, dist2, alternative='two-sided')
    
    # Jensen-Shannon divergence (symmetric KL divergence)
    # Requires binning
    bins = np.histogram_bin_edges(np.concatenate([dist1, dist2]), bins=50)
    p1, _ = np.histogram(dist1, bins=bins, density=True)
    p2, _ = np.histogram(dist2, bins=bins, density=True)
    
    # Add small epsilon to avoid log(0)
    p1 = p1 + 1e-10
    p2 = p2 + 1e-10
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    m = (p1 + p2) / 2
    js_divergence = 0.5 * stats.entropy(p1, m) + 0.5 * stats.entropy(p2, m)
    
    return {
        'kolmogorov_smirnov': {
            'statistic': ks_stat,
            'p_value': ks_pval,
            'significant': ks_pval < 0.05
        },
        'wasserstein_distance': emd,
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret_cohens_d(cohens_d),
        'mann_whitney': {
            'statistic': mw_stat,
            'p_value': mw_pval,
            'significant': mw_pval < 0.05
        },
        'jensen_shannon_divergence': js_divergence,
        'means': {
            labels[0]: np.mean(dist1),
            labels[1]: np.mean(dist2)
        },
        'medians': {
            labels[0]: np.median(dist1),
            labels[1]: np.median(dist2)
        }
    }

def interpret_cohens_d(d: float) -> str:
    """Standard interpretation of Cohen's d"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def plot_distribution_comparison(
    dist1: np.ndarray,
    dist2: np.ndarray,
    labels: Tuple[str, str] = ('Garden Path', 'Control'),
    title: str = 'Token Update Distributions'
) -> plt.Figure:
    """
    Visualize distributions without imposing thresholds.
    Multiple visualization methods for robustness.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Kernel Density Estimation
    ax = axes[0, 0]
    kde1 = gaussian_kde(dist1)
    kde2 = gaussian_kde(dist2)
    x_range = np.linspace(
        min(dist1.min(), dist2.min()),
        max(dist1.max(), dist2.max()),
        1000
    )
    
    ax.plot(x_range, kde1(x_range), label=labels[0], linewidth=2)
    ax.plot(x_range, kde2(x_range), label=labels[1], linewidth=2)
    ax.fill_between(x_range, kde1(x_range), alpha=0.3)
    ax.fill_between(x_range, kde2(x_range), alpha=0.3)
    ax.set_xlabel('Cosine Distance')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative Distribution Function
    ax = axes[0, 1]
    ax.plot(np.sort(dist1), np.linspace(0, 1, len(dist1)), 
            label=labels[0], linewidth=2)
    ax.plot(np.sort(dist2), np.linspace(0, 1, len(dist2)), 
            label=labels[1], linewidth=2)
    ax.set_xlabel('Cosine Distance')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Violin Plot
    ax = axes[1, 0]
    parts = ax.violinplot([dist1, dist2], positions=[0, 1], 
                          widths=0.7, showmeans=True, showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Cosine Distance')
    ax.set_title('Distribution Shape Comparison')
    ax.grid(True, alpha=0.3)
    
    # 4. Box Plot with outliers
    ax = axes[1, 1]
    bp = ax.boxplot([dist1, dist2], labels=labels, 
                    notch=True, patch_artist=True)
    ax.set_ylabel('Cosine Distance')
    ax.set_title('Quartiles and Outliers')
    ax.grid(True, alpha=0.3)
    
    # Color the box plots
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def threshold_robustness_analysis(
    dist1: np.ndarray,
    dist2: np.ndarray,
    thresholds: np.ndarray = None
) -> Dict:
    """
    Show that results don't depend on arbitrary threshold choice.
    This addresses reviewer concerns about threshold selection.
    """
    
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    
    results = []
    
    for thresh in thresholds:
        rate1 = (dist1 > thresh).mean()
        rate2 = (dist2 > thresh).mean()
        
        # Effect size at this threshold
        if rate1 + rate2 > 0:
            cohens_d = (rate1 - rate2) / np.sqrt((np.var([rate1]) + np.var([rate2])) / 2)
        else:
            cohens_d = 0
        
        results.append({
            'threshold': thresh,
            'rate_dist1': rate1,
            'rate_dist2': rate2,
            'difference': rate1 - rate2,
            'cohens_d': cohens_d
        })
    
    return pd.DataFrame(results)