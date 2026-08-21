#!/usr/bin/env python3
"""
Aggregate train_and_eval_3_fast outputs (test_fast/classes{n}/run{r}/test_results.npz)
into CSVs and PE-bias plots, mirroring analyze_v5_results.py + test/PE_test_3.py logic.

Run from anywhere; defaults to this script's directory as data root:
  python analyze_fast_results.py
  python analyze_fast_results.py --plots-only 55
  python analyze_fast_results.py --midpoint
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use("default")

BASE_DIR = Path(__file__).resolve().parent


def midpoint_target_fraction(n: int) -> float:
	"""Midpoint between chance accuracy (1/n) and perfect (1), as a fraction in [0, 1]."""
	return (1.0 + 1.0 / float(n)) / 2.0


def _natural_run_sort_key(path: str) -> int:
	m = re.search(r"/run(\d+)/", path.replace("\\", "/"))
	return int(m.group(1)) if m else 0


def aggregate_runs_for_n_classes(base_dir: Path, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
	"""Load all test_results.npz for one n_classes; return stacked arrays and signal/noise grids."""
	pattern = str(base_dir / f"classes{n_classes}" / "run*" / "test_results.npz")
	files = sorted(glob.glob(pattern), key=_natural_run_sort_key)
	if not files:
		return None

	all_runs_test_acc = []
	all_runs_test_conf = []
	signal_test_vals = noise_test_vals = None

	for fp in files:
		z = np.load(fp)
		if signal_test_vals is None:
			signal_test_vals = z["signal_test_vals"]
			noise_test_vals = z["noise_test_vals"]
		all_runs_test_acc.append(z["all_test_acc"] / 100.0)
		all_runs_test_conf.append(z["all_test_conf"] / 100.0)

	all_runs_test_acc = np.array(all_runs_test_acc)
	all_runs_test_conf = np.array(all_runs_test_conf)
	return all_runs_test_acc, all_runs_test_conf, signal_test_vals, noise_test_vals, np.array(files)


def build_pe_dataframe(
	all_runs_test_acc: np.ndarray,
	all_runs_test_conf: np.ndarray,
	signal_test_vals: np.ndarray,
	noise_test_vals: np.ndarray,
	n_classes: int,
	target_acc_pct: int | None = None,
	*,
	target_acc_frac: float | None = None,
) -> pd.DataFrame:
	"""Match test/PE_test_3.py: PE points = argmin |mean_acc[noise_row] - target| over signal."""
	if target_acc_frac is not None:
		target_acc = float(target_acc_frac)
	elif target_acc_pct is not None:
		target_acc = target_acc_pct / 100.0
	else:
		raise ValueError("build_pe_dataframe requires target_acc_pct or target_acc_frac")
	all_runs_test_acc_mean = all_runs_test_acc.mean(0)
	all_runs_test_conf_mean = all_runs_test_conf.mean(0)

	low_PE_ind = np.abs(all_runs_test_acc_mean[0, :] - target_acc).argmin()
	high_PE_ind = np.abs(all_runs_test_acc_mean[1, :] - target_acc).argmin()
	low_PE_signal = signal_test_vals[low_PE_ind]
	high_PE_signal = signal_test_vals[high_PE_ind]

	n_runs = all_runs_test_acc.shape[0]
	low_PE_test_acc = all_runs_test_acc[:, 0, low_PE_ind]
	high_PE_test_acc = all_runs_test_acc[:, 1, high_PE_ind]
	low_PE_test_conf = all_runs_test_conf[:, 0, low_PE_ind]
	high_PE_test_conf = all_runs_test_conf[:, 1, high_PE_ind]

	return pd.DataFrame(
		{
			"low_PE_test_acc": low_PE_test_acc,
			"low_PE_test_conf": low_PE_test_conf,
			"high_PE_test_acc": high_PE_test_acc,
			"high_PE_test_conf": high_PE_test_conf,
			"low_PE_signal": np.repeat(low_PE_signal, n_runs),
			"high_PE_signal": np.repeat(high_PE_signal, n_runs),
			"low_PE_noise": np.repeat(noise_test_vals[0], n_runs),
			"high_PE_noise": np.repeat(noise_test_vals[1], n_runs),
			"dimensionality": np.repeat(n_classes, n_runs),
		}
	)


def write_csvs_for_all_n(base_dir: Path, target_accs: list[int]) -> dict[int, dict[int, pd.DataFrame]]:
	data_by_classes: dict[int, dict[int, pd.DataFrame]] = {}
	for n_classes in range(2, 11):
		packed = aggregate_runs_for_n_classes(base_dir, n_classes)
		if packed is None:
			print(f"No npz files under {base_dir}/classes{n_classes}/run*/")
			continue
		all_runs_test_acc, all_runs_test_conf, signal_test_vals, noise_test_vals, files = packed
		print(f"classes{n_classes}: {len(files)} runs")
		data_by_classes[n_classes] = {}
		for ta in target_accs:
			df = build_pe_dataframe(
				all_runs_test_acc,
				all_runs_test_conf,
				signal_test_vals,
				noise_test_vals,
				n_classes,
				target_acc_pct=ta,
			)
			out = base_dir / f"df{n_classes}_fast_ta{ta}.csv"
			df.to_csv(out, index=False)
			data_by_classes[n_classes][ta] = df
			print(f"  wrote {out.name}")
	return data_by_classes


def load_fast_csvs(base_dir: Path, target_accs: list[int]) -> dict[int, dict[int, pd.DataFrame]]:
	"""Load pre-built CSVs (same layout as analyze_v5_results expects for v5)."""
	data_by_classes: dict[int, dict[int, pd.DataFrame]] = {}
	for n_classes in range(2, 11):
		data_by_classes[n_classes] = {}
		for ta in target_accs:
			path = base_dir / f"df{n_classes}_fast_ta{ta}.csv"
			if path.is_file():
				data_by_classes[n_classes][ta] = pd.read_csv(path)
	return data_by_classes


def write_midpoint_csvs(base_dir: Path) -> dict[int, dict[str, pd.DataFrame]]:
	"""Per n_classes, PE target = (1 + 1/n) / 2 (midpoint between random guessing and 100%)."""
	data_by_classes: dict[int, dict[str, pd.DataFrame]] = {}
	for n_classes in range(2, 11):
		packed = aggregate_runs_for_n_classes(base_dir, n_classes)
		if packed is None:
			print(f"No npz files under {base_dir}/classes{n_classes}/run*/")
			continue
		all_runs_test_acc, all_runs_test_conf, signal_test_vals, noise_test_vals, files = packed
		ta_frac = midpoint_target_fraction(n_classes)
		print(f"classes{n_classes}: {len(files)} runs, midpoint target acc = {ta_frac:.4f} ({100 * ta_frac:.2f}%)")
		df = build_pe_dataframe(
			all_runs_test_acc,
			all_runs_test_conf,
			signal_test_vals,
			noise_test_vals,
			n_classes,
			target_acc_frac=ta_frac,
		)
		df["midpoint_target_accuracy"] = ta_frac
		out = base_dir / f"df{n_classes}_fast_midpoint.csv"
		df.to_csv(out, index=False)
		data_by_classes[n_classes] = {"midpoint": df}
		print(f"  wrote {out.name}")
	return data_by_classes


def load_midpoint_csvs(base_dir: Path) -> dict[int, dict[str, pd.DataFrame]]:
	data_by_classes: dict[int, dict[str, pd.DataFrame]] = {}
	for n_classes in range(2, 11):
		path = base_dir / f"df{n_classes}_fast_midpoint.csv"
		if path.is_file():
			data_by_classes[n_classes] = {"midpoint": pd.read_csv(path)}
	return data_by_classes


def plot_pe_bias_by_classes(data_by_classes: dict, target_acc: int, out_dir: Path) -> None:
	"""Copied from analyze_v5_results.plot_pe_bias_by_classes (same error-bar behavior as v5)."""
	n_classes_list = []
	low_pe_conf = []
	high_pe_conf = []
	conf_diff = []
	conf_diff_se = []

	for n_classes in range(2, 11):
		if n_classes in data_by_classes and target_acc in data_by_classes[n_classes]:
			df = data_by_classes[n_classes][target_acc]
			n_classes_list.append(n_classes)
			low_pe_conf.append(df["low_PE_test_conf"].mean())
			high_pe_conf.append(df["high_PE_test_conf"].mean())
			conf_diff.append(df["high_PE_test_conf"].mean() - df["low_PE_test_conf"].mean())
			conf_diff_se.append(df["high_PE_test_conf"].std() / np.sqrt(len(df)))

	if not n_classes_list:
		print(f"No data to plot for target_acc={target_acc}")
		return

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

	ax1.errorbar(n_classes_list, low_pe_conf, yerr=conf_diff_se, marker="o", label="Low PE", capsize=5, linewidth=2)
	ax1.errorbar(n_classes_list, high_pe_conf, yerr=conf_diff_se, marker="s", label="High PE", capsize=5, linewidth=2)
	ax1.set_xlabel("Number of Classes (n_classes)")
	ax1.set_ylabel("Confidence")
	ax1.set_title("Confidence by PE Condition")
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	ax2.errorbar(n_classes_list, conf_diff, yerr=conf_diff_se, marker="D", color="red", capsize=5, linewidth=2)
	ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
	ax2.set_xlabel("Number of Classes (n_classes)")
	ax2.set_ylabel("PE Bias (High PE - Low PE)")
	ax2.set_title("PE Bias vs Number of Classes")
	ax2.grid(True, alpha=0.3)

	plt.suptitle(
		f"PE Bias Analysis (train_and_eval_3_fast, Target Acc = {target_acc}%)",
		fontsize=16,
	)
	plt.tight_layout()
	out_path = out_dir / f"pe_bias_by_classes_fast_ta{target_acc}.png"
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")


def plot_pe_bias_midpoint(data_by_classes: dict, out_dir: Path) -> None:
	"""Same layout as plot_pe_bias_by_classes; each n used its own midpoint target when building CSVs."""
	n_classes_list = []
	low_pe_conf = []
	high_pe_conf = []
	conf_diff = []
	conf_diff_se = []

	for n_classes in range(2, 11):
		if n_classes in data_by_classes and "midpoint" in data_by_classes[n_classes]:
			df = data_by_classes[n_classes]["midpoint"]
			n_classes_list.append(n_classes)
			low_pe_conf.append(df["low_PE_test_conf"].mean())
			high_pe_conf.append(df["high_PE_test_conf"].mean())
			conf_diff.append(df["high_PE_test_conf"].mean() - df["low_PE_test_conf"].mean())
			conf_diff_se.append(df["high_PE_test_conf"].std() / np.sqrt(len(df)))

	if not n_classes_list:
		print("No midpoint data to plot")
		return

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

	ax1.errorbar(n_classes_list, low_pe_conf, yerr=conf_diff_se, marker="o", label="Low PE", capsize=5, linewidth=2)
	ax1.errorbar(n_classes_list, high_pe_conf, yerr=conf_diff_se, marker="s", label="High PE", capsize=5, linewidth=2)
	ax1.set_xlabel("Number of Classes (n_classes)")
	ax1.set_ylabel("Confidence")
	ax1.set_title("Confidence by PE Condition")
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	ax2.errorbar(n_classes_list, conf_diff, yerr=conf_diff_se, marker="D", color="red", capsize=5, linewidth=2)
	ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
	ax2.set_xlabel("Number of Classes (n_classes)")
	ax2.set_ylabel("PE Bias (High PE - Low PE)")
	ax2.set_title("PE Bias vs Number of Classes")
	ax2.grid(True, alpha=0.3)

	plt.suptitle(
		"PE Bias Analysis (train_and_eval_3_fast, midpoint target per n: (1 + 1/n) / 2)",
		fontsize=16,
	)
	plt.tight_layout()
	out_path = out_dir / "pe_bias_by_classes_fast_midpoint.png"
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")


def statistical_analysis(data_by_classes: dict, target_acc: int) -> None:
	print(f"\n=== STATISTICAL ANALYSIS (train_and_eval_3_fast, Target Acc = {target_acc}%) ===\n")
	for n_classes in range(2, 11):
		if n_classes not in data_by_classes or target_acc not in data_by_classes[n_classes]:
			continue
		df = data_by_classes[n_classes][target_acc]
		_, conf_p = stats.ttest_rel(df["high_PE_test_conf"], df["low_PE_test_conf"])
		conf_diff = df["high_PE_test_conf"].mean() - df["low_PE_test_conf"].mean()
		pooled_var = (df["high_PE_test_conf"].var() + df["low_PE_test_conf"].var()) / 2
		conf_cohen_d = conf_diff / np.sqrt(pooled_var) if pooled_var > 0 else float("nan")
		print(
			f"n_classes = {n_classes:2d}: Conf Diff = {conf_diff:6.4f}, p = {conf_p:6.4f}, "
			f"Cohen's d = {conf_cohen_d:6.4f}, n = {len(df):2d}"
		)


def statistical_analysis_midpoint(data_by_classes: dict) -> None:
	print("\n=== STATISTICAL ANALYSIS (train_and_eval_3_fast, midpoint targets (1+1/n)/2) ===\n")
	for n_classes in range(2, 11):
		if n_classes not in data_by_classes or "midpoint" not in data_by_classes[n_classes]:
			continue
		df = data_by_classes[n_classes]["midpoint"]
		ta_frac = float(df["midpoint_target_accuracy"].iloc[0]) if "midpoint_target_accuracy" in df.columns else midpoint_target_fraction(n_classes)
		_, conf_p = stats.ttest_rel(df["high_PE_test_conf"], df["low_PE_test_conf"])
		conf_diff = df["high_PE_test_conf"].mean() - df["low_PE_test_conf"].mean()
		pooled_var = (df["high_PE_test_conf"].var() + df["low_PE_test_conf"].var()) / 2
		conf_cohen_d = conf_diff / np.sqrt(pooled_var) if pooled_var > 0 else float("nan")
		print(
			f"n_classes = {n_classes:2d}: target = {ta_frac:.4f} ({100 * ta_frac:.2f}%) | "
			f"Conf Diff = {conf_diff:6.4f}, p = {conf_p:6.4f}, Cohen's d = {conf_cohen_d:6.4f}, n = {len(df):2d}"
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="PE bias plots from test_fast npz results")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=BASE_DIR,
		help="Directory containing classes2/, classes3/, ... (default: this script's folder)",
	)
	parser.add_argument(
		"--midpoint",
		action="store_true",
		help="Use target accuracy (1+1/n)/2 per class count n (between chance 1/n and 100%%); writes df{n}_fast_midpoint.csv and pe_bias_by_classes_fast_midpoint.png",
	)
	parser.add_argument(
		"--skip-csv",
		action="store_true",
		help="Only plot from existing CSVs (do not rebuild from npz)",
	)
	parser.add_argument(
		"--plots-only",
		type=int,
		metavar="TA",
		default=None,
		help="Only generate pe_bias plot for this target accuracy (default: all of 55,60,65,70,75)",
	)
	args = parser.parse_args()
	base_dir = args.data_dir.resolve()
	if not base_dir.is_dir():
		print(f"Data directory not found: {base_dir}", file=sys.stderr)
		sys.exit(1)

	if args.midpoint and args.plots_only is not None:
		parser.error("--midpoint cannot be combined with --plots-only")

	if args.midpoint:
		if args.skip_csv:
			data_by_classes = load_midpoint_csvs(base_dir)
		else:
			data_by_classes = write_midpoint_csvs(base_dir)
		plot_pe_bias_midpoint(data_by_classes, base_dir)
		statistical_analysis_midpoint(data_by_classes)
		return

	all_tas = [55, 60, 65, 70, 75]
	if args.skip_csv:
		data_by_classes = load_fast_csvs(base_dir, all_tas)
	else:
		data_by_classes = write_csvs_for_all_n(base_dir, all_tas)

	plot_tas = [args.plots_only] if args.plots_only is not None else all_tas
	for ta in plot_tas:
		has = any(ta in data_by_classes.get(n, {}) for n in range(2, 11))
		if has:
			plot_pe_bias_by_classes(data_by_classes, ta, base_dir)
			statistical_analysis(data_by_classes, ta)
		else:
			print(f"No data for target accuracy {ta}%")


if __name__ == "__main__":
	main()
