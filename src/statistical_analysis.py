"""
Statistical Analysis Module for Experimental Results
====================================================

Implements:
1. Statistical significance tests (t-test, Wilcoxon, Mann-Whitney)
2. Effect size calculations (Cohen's d, Cliff's delta)
3. Multiple comparison corrections (Bonferroni, Holm)
4. Confidence intervals
5. Result visualization

Author: Advanced GNN Research
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


@dataclass
class StatisticalTestResult:
    """Container for statistical test results"""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for experimental results
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha

    def paired_t_test(
        self,
        method_a_results: List[float],
        method_b_results: List[float],
        alternative: str = "two-sided",
    ) -> StatisticalTestResult:
        """
        Paired t-test for comparing two methods on same data

        Args:
            method_a_results: Results from method A
            method_b_results: Results from method B
            alternative: 'two-sided', 'less', or 'greater'
        """
        method_a = np.array(method_a_results)
        method_b = np.array(method_b_results)

        # Paired t-test
        statistic, p_value = stats.ttest_rel(method_a, method_b, alternative=alternative)

        # Effect size (Cohen's d for paired samples)
        diff = method_a - method_b
        effect_size = np.mean(diff) / np.std(diff, ddof=1)

        # Confidence interval for mean difference
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff)
        ci = stats.t.interval(0.95, len(diff) - 1, loc=mean_diff, scale=se_diff)

        # Significance
        significant = p_value < self.alpha

        # Interpretation
        if significant:
            if method_a.mean() > method_b.mean():
                interpretation = f"Method A significantly outperforms Method B (p={p_value:.4f})"
            else:
                interpretation = f"Method B significantly outperforms Method A (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
        )

    def wilcoxon_test(
        self,
        method_a_results: List[float],
        method_b_results: List[float],
        alternative: str = "two-sided",
    ) -> StatisticalTestResult:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test)

        More robust to outliers and non-normal distributions
        """
        method_a = np.array(method_a_results)
        method_b = np.array(method_b_results)

        # Wilcoxon test
        statistic, p_value = stats.wilcoxon(
            method_a, method_b, alternative=alternative, zero_method="zsplit"
        )

        # Effect size (rank-biserial correlation)
        diff = method_a - method_b
        n = len(diff)
        r = 1 - (2 * statistic) / (n * (n + 1) / 2)
        effect_size = r

        # Significance
        significant = p_value < self.alpha

        # Interpretation
        if significant:
            if np.median(method_a) > np.median(method_b):
                interpretation = (
                    f"Method A significantly better (p={p_value:.4f}, rank-biserial r={r:.3f})"
                )
            else:
                interpretation = (
                    f"Method B significantly better (p={p_value:.4f}, rank-biserial r={r:.3f})"
                )
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=(np.nan, np.nan),  # Not directly available
            interpretation=interpretation,
        )

    def mann_whitney_test(
        self,
        method_a_results: List[float],
        method_b_results: List[float],
        alternative: str = "two-sided",
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U test (unpaired non-parametric test)

        Use when samples are independent (not paired)
        """
        method_a = np.array(method_a_results)
        method_b = np.array(method_b_results)

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(method_a, method_b, alternative=alternative)

        # Effect size (rank-biserial correlation for Mann-Whitney)
        n1, n2 = len(method_a), len(method_b)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        # Significance
        significant = p_value < self.alpha

        # Interpretation
        if significant:
            if np.median(method_a) > np.median(method_b):
                interpretation = f"Method A significantly better (p={p_value:.4f})"
            else:
                interpretation = f"Method B significantly better (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=(np.nan, np.nan),
            interpretation=interpretation,
        )

    def cohens_d(
        self, method_a_results: List[float], method_b_results: List[float], paired: bool = True
    ) -> float:
        """
        Calculate Cohen's d effect size

        Interpretation:
        - Small: d ≈ 0.2
        - Medium: d ≈ 0.5
        - Large: d ≈ 0.8
        """
        method_a = np.array(method_a_results)
        method_b = np.array(method_b_results)

        if paired:
            # For paired samples
            diff = method_a - method_b
            d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            # For independent samples
            mean_diff = np.mean(method_a) - np.mean(method_b)
            pooled_std = np.sqrt(
                (
                    (len(method_a) - 1) * np.var(method_a, ddof=1)
                    + (len(method_b) - 1) * np.var(method_b, ddof=1)
                )
                / (len(method_a) + len(method_b) - 2)
            )
            d = mean_diff / pooled_std

        return d

    def cliffs_delta(self, method_a_results: List[float], method_b_results: List[float]) -> float:
        """
        Calculate Cliff's delta (non-parametric effect size)

        Range: [-1, 1]
        - Negative: Method B is better
        - Positive: Method A is better

        Interpretation:
        - Negligible: |d| < 0.147
        - Small: |d| < 0.33
        - Medium: |d| < 0.474
        - Large: |d| >= 0.474
        """
        method_a = np.array(method_a_results)
        method_b = np.array(method_b_results)

        n1, n2 = len(method_a), len(method_b)

        # Count dominance
        dominance = 0
        for a in method_a:
            for b in method_b:
                if a > b:
                    dominance += 1
                elif a < b:
                    dominance -= 1

        delta = dominance / (n1 * n2)
        return delta

    def bonferroni_correction(
        self, p_values: List[float], alpha: float = None
    ) -> Tuple[List[bool], float]:
        """
        Bonferroni correction for multiple comparisons

        Returns:
            significant: List of boolean indicating significance
            corrected_alpha: Adjusted alpha level
        """
        if alpha is None:
            alpha = self.alpha

        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests
        significant = [p < corrected_alpha for p in p_values]

        return significant, corrected_alpha

    def holm_correction(self, p_values: List[float], alpha: float = None) -> List[bool]:
        """
        Holm-Bonferroni correction (less conservative than Bonferroni)
        """
        if alpha is None:
            alpha = self.alpha

        n_tests = len(p_values)

        # Sort p-values with indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]

        # Test each p-value
        significant = np.zeros(n_tests, dtype=bool)

        for i, p in enumerate(sorted_p_values):
            adjusted_alpha = alpha / (n_tests - i)
            if p < adjusted_alpha:
                significant[sorted_indices[i]] = True
            else:
                break  # Stop testing once we fail

        return significant.tolist()

    def confidence_interval(
        self, data: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean
        """
        data = np.array(data)
        mean = np.mean(data)
        se = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=se)
        return ci

    def compare_multiple_methods(
        self, results_dict: Dict[str, List[float]], reference_method: str = None
    ) -> Dict[str, Dict]:
        """
        Compare multiple methods with statistical tests

        Args:
            results_dict: {method_name: [results]}
            reference_method: Method to compare against (default: best performing)

        Returns:
            comparison_results: Detailed statistical comparison
        """
        # Determine reference method
        if reference_method is None:
            # Use method with highest mean as reference
            reference_method = max(results_dict.items(), key=lambda x: np.mean(x[1]))[0]

        reference_results = results_dict[reference_method]

        comparison_results = {}
        p_values = []

        for method_name, method_results in results_dict.items():
            if method_name == reference_method:
                continue

            # Paired t-test
            t_test = self.paired_t_test(reference_results, method_results)

            # Wilcoxon test
            wilcoxon = self.wilcoxon_test(reference_results, method_results)

            # Effect sizes
            cohens = self.cohens_d(reference_results, method_results, paired=True)
            cliffs = self.cliffs_delta(reference_results, method_results)

            comparison_results[method_name] = {
                "t_test": t_test,
                "wilcoxon": wilcoxon,
                "cohens_d": cohens,
                "cliffs_delta": cliffs,
                "mean_diff": np.mean(reference_results) - np.mean(method_results),
                "median_diff": np.median(reference_results) - np.median(method_results),
            }

            p_values.append(t_test.p_value)

        # Multiple comparison correction
        if len(p_values) > 0:
            bonferroni_sig, bonf_alpha = self.bonferroni_correction(p_values)
            holm_sig = self.holm_correction(p_values)

            for i, method_name in enumerate(
                [m for m in results_dict.keys() if m != reference_method]
            ):
                comparison_results[method_name]["bonferroni_significant"] = bonferroni_sig[i]
                comparison_results[method_name]["holm_significant"] = holm_sig[i]

        return {
            "reference_method": reference_method,
            "comparisons": comparison_results,
            "corrected_alpha_bonferroni": bonf_alpha if len(p_values) > 0 else self.alpha,
        }

    def generate_summary_table(self, results_dict: Dict[str, List[float]]) -> str:
        """
        Generate a formatted summary table of results
        """
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL SUMMARY")
        lines.append("=" * 80)
        lines.append(f"{'Method':<20} {'Mean':<12} {'Std':<12} {'Median':<12} {'95% CI':<20}")
        lines.append("-" * 80)

        for method_name, results in sorted(
            results_dict.items(), key=lambda x: np.mean(x[1]), reverse=True
        ):
            results = np.array(results)
            mean = np.mean(results)
            std = np.std(results, ddof=1)
            median = np.median(results)
            ci = self.confidence_interval(results)

            lines.append(
                f"{method_name:<20} "
                f"{mean:<12.4f} "
                f"{std:<12.4f} "
                f"{median:<12.4f} "
                f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


class ResultVisualizer:
    """
    Visualize statistical comparison results
    """

    @staticmethod
    def plot_comparison_boxplot(
        results_dict: Dict[str, List[float]],
        title: str = "Method Comparison",
        ylabel: str = "Performance",
        save_path: Optional[str] = None,
    ):
        """
        Create boxplot comparing multiple methods
        """
        plt.figure(figsize=(12, 6))

        # Prepare data
        methods = list(results_dict.keys())
        data = [results_dict[m] for m in methods]

        # Create boxplot
        bp = plt.boxplot(data, labels=methods, patch_artist=True)

        # Color boxes
        colors = sns.color_palette("Set2", len(methods))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        # Highlight best method
        means = [np.mean(d) for d in data]
        best_idx = np.argmax(means)
        bp["boxes"][best_idx].set_facecolor("gold")
        bp["boxes"][best_idx].set_edgecolor("red")
        bp["boxes"][best_idx].set_linewidth(2)

        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, weight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_pairwise_comparison(comparison_results: Dict, save_path: Optional[str] = None):
        """
        Visualize pairwise comparison results
        """
        reference = comparison_results["reference_method"]
        comparisons = comparison_results["comparisons"]

        methods = list(comparisons.keys())
        p_values = [comparisons[m]["t_test"].p_value for m in methods]
        effect_sizes = [comparisons[m]["cohens_d"] for m in methods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # P-values
        colors = ["red" if p < 0.05 else "gray" for p in p_values]
        ax1.barh(methods, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
        ax1.axvline(-np.log10(0.05), color="red", linestyle="--", label="α=0.05")
        ax1.set_xlabel("-log10(p-value)", fontsize=11)
        ax1.set_title(f"Statistical Significance vs {reference}", fontsize=12, weight="bold")
        ax1.legend()
        ax1.grid(axis="x", alpha=0.3)

        # Effect sizes
        colors = [
            "green" if abs(d) > 0.5 else "orange" if abs(d) > 0.2 else "gray" for d in effect_sizes
        ]
        ax2.barh(methods, effect_sizes, color=colors, alpha=0.7)
        ax2.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax2.axvline(0.2, color="orange", linestyle="--", alpha=0.5, label="Small")
        ax2.axvline(0.5, color="green", linestyle="--", alpha=0.5, label="Medium")
        ax2.axvline(0.8, color="darkgreen", linestyle="--", alpha=0.5, label="Large")
        ax2.set_xlabel("Cohen's d", fontsize=11)
        ax2.set_title("Effect Size", fontsize=12, weight="bold")
        ax2.legend()
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


# Test and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("Statistical Analysis Module")
    print("=" * 80)

    # Generate sample data
    np.random.seed(42)
    method_a = np.random.normal(0.75, 0.1, 20)
    method_b = np.random.normal(0.65, 0.12, 20)
    method_c = np.random.normal(0.70, 0.08, 20)

    results = {
        "Our Method": method_a.tolist(),
        "Baseline 1": method_b.tolist(),
        "Baseline 2": method_c.tolist(),
    }

    # Analyze
    analyzer = StatisticalAnalyzer(alpha=0.05)

    print("\n" + analyzer.generate_summary_table(results))

    print("\nPairwise Comparisons:")
    comparison = analyzer.compare_multiple_methods(results)

    for method, stats in comparison["comparisons"].items():
        print(f"\n{method}:")
        print(f"  t-test: {stats['t_test'].interpretation}")
        print(f"  Cohen's d: {stats['cohens_d']:.3f}")
        print(f"  Cliff's delta: {stats['cliffs_delta']:.3f}")
        print(f"  Bonferroni significant: {stats['bonferroni_significant']}")

    print("\n✓ Statistical analysis complete!")
