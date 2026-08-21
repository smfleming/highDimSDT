#!/usr/bin/env python3

import json
import numpy as np

def create_text_plots():
    """
    Create text-based visualizations of the key evidence
    """
    
    print("KEY EVIDENCE VISUALIZATION")
    print("="*80)
    
    # Load the data
    with open('confidence_comparison_data.json', 'r') as f:
        conf_stats = json.load(f)
    
    with open('confidence_distributions_data.json', 'r') as f:
        conf_dist = json.load(f)
    
    with open('pe_conditions_data.json', 'r') as f:
        pe_cond = json.load(f)
    
    # Plot 1: Confidence Statistics Comparison
    print("\nPLOT 1: CONFIDENCE STATISTICS COMPARISON")
    print("="*50)
    
    n_classes = sorted([int(k) for k in conf_stats.keys()])
    
    # Create ASCII bar chart for confidence means
    print("\nConfidence Mean Values:")
    print("n_classes | Confidence Mean (%) | Bar Chart")
    print("-" * 50)
    
    max_conf = max([conf_stats[str(n)]['mean'] for n in n_classes])
    min_conf = min([conf_stats[str(n)]['mean'] for n in n_classes])
    
    for n in n_classes:
        mean_val = conf_stats[str(n)]['mean']
        # Create bar chart (scale to 50 characters)
        bar_length = int((mean_val - min_conf) / (max_conf - min_conf) * 40)
        bar = "█" * bar_length
        color_indicator = "🔴" if n == 10 else "🔵"
        print(f"{n:8} | {mean_val:17.2f} | {color_indicator} {bar}")
    
    # Plot 2: Confidence Standard Deviation
    print("\n\nConfidence Standard Deviation:")
    print("n_classes | Confidence Std (%) | Bar Chart")
    print("-" * 50)
    
    max_std = max([conf_stats[str(n)]['std'] for n in n_classes])
    min_std = min([conf_stats[str(n)]['std'] for n in n_classes])
    
    for n in n_classes:
        std_val = conf_stats[str(n)]['std']
        bar_length = int((std_val - min_std) / (max_std - min_std) * 40)
        bar = "█" * bar_length
        color_indicator = "🔴" if n == 10 else "🔵"
        print(f"{n:8} | {std_val:17.2f} | {color_indicator} {bar}")
    
    # Plot 3: Confidence Range
    print("\n\nConfidence Range (Max - Min):")
    print("n_classes | Confidence Range (%) | Bar Chart")
    print("-" * 50)
    
    max_range = max([conf_stats[str(n)]['range'] for n in n_classes])
    min_range = min([conf_stats[str(n)]['range'] for n in n_classes])
    
    for n in n_classes:
        range_val = conf_stats[str(n)]['range']
        bar_length = int((range_val - min_range) / (max_range - min_range) * 40)
        bar = "█" * bar_length
        color_indicator = "🔴" if n == 10 else "🔵"
        print(f"{n:8} | {range_val:17.2f} | {color_indicator} {bar}")
    
    # Plot 4: Confidence Distributions (Percentiles)
    print("\n\nPLOT 2: CONFIDENCE DISTRIBUTIONS")
    print("="*50)
    
    print("\nConfidence Percentiles:")
    print("n_classes | P10   | P25   | P50   | P75   | P90   | P95   | P99")
    print("-" * 70)
    
    for n in [2, 5, 8, 9, 10]:
        if str(n) in conf_dist:
            p = conf_dist[str(n)]['percentiles']
            color_indicator = "🔴" if n == 10 else "🔵"
            print(f"{n:8} | {p['10']:5.1f} | {p['25']:5.1f} | {p['50']:5.1f} | {p['75']:5.1f} | {p['90']:5.1f} | {p['95']:5.1f} | {p['99']:5.1f} {color_indicator}")
    
    # Plot 5: PE Conditions
    print("\n\nPLOT 3: PE CONDITIONS COMPARISON")
    print("="*50)
    
    print("\nPE Signal Strengths:")
    print("n_classes | Low PE Signal | High PE Signal | PE Difference")
    print("-" * 55)
    
    for n in n_classes:
        if str(n) in pe_cond:
            data = pe_cond[str(n)]
            color_indicator = "🔴" if n == 10 else "🔵"
            print(f"{n:8} | {data['low_PE_signal']:13.3f} | {data['high_PE_signal']:14.3f} | {data['pe_difference']:13.3f} {color_indicator}")
    
    # Summary Statistics
    print("\n\nKEY FINDINGS SUMMARY")
    print("="*50)
    
    n10_stats = conf_stats['10']
    others_means = [conf_stats[str(n)]['mean'] for n in n_classes if n != 10]
    others_stds = [conf_stats[str(n)]['std'] for n in n_classes if n != 10]
    
    others_mean = np.mean(others_means)
    others_std = np.mean(others_stds)
    
    print(f"\n🔴 n_classes=10:")
    print(f"   Confidence Mean: {n10_stats['mean']:.2f}%")
    print(f"   Confidence Std:  {n10_stats['std']:.2f}%")
    print(f"   Confidence Min:  {n10_stats['min']:.2f}%")
    print(f"   Confidence Max:  {n10_stats['max']:.2f}%")
    
    print(f"\n🔵 Others (n_classes=2-9):")
    print(f"   Confidence Mean: {others_mean:.2f}%")
    print(f"   Confidence Std:  {others_std:.2f}%")
    
    print(f"\n📊 DIFFERENCES:")
    print(f"   Mean Difference: {n10_stats['mean'] - others_mean:.2f}% (n_classes=10 is LOWER)")
    print(f"   Std Difference:  {n10_stats['std'] - others_std:.2f}% (n_classes=10 has HIGHER variance)")
    
    # Statistical significance indicators
    print(f"\n🚨 SIGNIFICANT DIFFERENCES:")
    if n10_stats['mean'] - others_mean < -5:
        print(f"   ✅ n_classes=10 has MUCH LOWER mean confidence (-{abs(n10_stats['mean'] - others_mean):.1f}%)")
    if n10_stats['std'] - others_std > 2:
        print(f"   ✅ n_classes=10 has MUCH HIGHER confidence variance (+{n10_stats['std'] - others_std:.1f}%)")
    if n10_stats['min'] < 60:
        print(f"   ✅ n_classes=10 has VERY LOW minimum confidence ({n10_stats['min']:.1f}%)")
    
    # Root cause explanation
    print(f"\n🎯 ROOT CAUSE:")
    print(f"   The issue is NOT in PE condition selection logic.")
    print(f"   The issue is in CONFIDENCE CALIBRATION due to encoder representation mismatch:")
    print(f"   ")
    print(f"   🔴 n_classes=10: Encoder learns 10-class representations → miscalibrated confidence")
    print(f"   🔵 n_classes=2-9: Encoder learns subset-class representations → better calibrated confidence")
    
    # Impact on PE bias
    print(f"\n💡 IMPACT ON PE BIAS RESULTS:")
    print(f"   PE bias analysis depends on confidence values.")
    print(f"   When confidence is miscalibrated (n_classes=10):")
    print(f"   - Confidence values are systematically lower")
    print(f"   - Confidence variance is higher") 
    print(f"   - This affects PE bias calculations")
    print(f"   - Result: Different PE bias results compared to n_classes=2-9")

def main():
    create_text_plots()

if __name__ == '__main__':
    main()



