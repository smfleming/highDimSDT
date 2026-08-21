import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from architecture import *
from util import log


def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)


class IndexedSubset(torch.utils.data.Dataset):
	"""Same samples/labels as torch.utils.data.Subset; faster index storage and __getitem__."""

	def __init__(self, dataset, indices):
		self.dataset = dataset
		self.indices = np.asarray(indices, dtype=np.int64)

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		return self.dataset[int(self.indices[idx])]


def _targets_as_numpy(dataset):
	t = dataset.targets
	return t.detach().numpy() if hasattr(t, "detach") else (t.numpy() if hasattr(t, "numpy") else np.asarray(t))


def train(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch, non_blocking):
	# Create file for saving training progress
	train_prog_dir = "./train_prog_fast/"
	check_path(train_prog_dir)
	model_dir = train_prog_dir + "run" + str(args.run) + "/"
	check_path(model_dir)
	train_prog_fname = model_dir + "epoch_" + str(epoch) + ".txt"
	train_prog_f = open(train_prog_fname, "w")
	train_prog_f.write("batch class_loss class_acc conf_loss conf\n")
	# Set to training mode
	encoder.train()
	class_out.train()
	conf_out.train()
	hardtanh = nn.Hardtanh()
	class_loss_fn = torch.nn.CrossEntropyLoss()
	conf_loss_fn = torch.nn.BCELoss()
	# Iterate over batches
	for batch_idx, (data, target) in enumerate(train_loader):
		# Batch start time
		start_time = time.time()
		# Load data
		x = data.to(device, non_blocking=non_blocking)
		y_targ = target.to(device, non_blocking=non_blocking)
		# Scale signal
		signal = (torch.rand(x.shape[0], device=device) * (args.signal_range[1] - args.signal_range[0])) + args.signal_range[0]
		x = x * signal.view(-1, 1, 1, 1)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		noise = (torch.rand(x.shape[0], device=device) * (args.noise_range[1] - args.noise_range[0])) + args.noise_range[0]
		noise = noise.view(-1, 1, 1, 1)
		x = x + torch.randn_like(x) * noise
		# Threshold image
		x = hardtanh(x)
		# Zero out gradients for optimizer
		optimizer.zero_grad()
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z)
		conf = conf_out(z)
		# Classification loss
		class_loss = class_loss_fn(y_pred, y_targ)
		# Classification accuracy
		_, y_pred_argmax = y_pred.max(1)
		correct_preds = torch.eq(y_pred_argmax, y_targ).type(torch.float)
		class_acc = correct_preds.mean().item() * 100.0
		# Confidence loss
		conf_loss = conf_loss_fn(conf.squeeze(), correct_preds)
		# Combine losses and update model
		combined_loss = class_loss + conf_loss
		combined_loss.backward()
		optimizer.step()
		# Overall confidence
		avg_conf = conf.mean().item() * 100.0
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report prgoress
		if batch_idx % args.log_interval == 0:
			log.info(
				"[Epoch: "
				+ str(epoch)
				+ "] "
				+ "[Batch: "
				+ str(batch_idx)
				+ " of "
				+ str(len(train_loader))
				+ "] "
				+ "[Class. Loss = "
				+ "{:.4f}".format(class_loss.item())
				+ "] "
				+ "[Class. Acc. = "
				+ "{:.2f}".format(class_acc)
				+ "] "
				+ "[Conf. Loss = "
				+ "{:.4f}".format(conf_loss.item())
				+ "] "
				+ "[Conf. = "
				+ "{:.2f}".format(avg_conf)
				+ "] "
				+ "["
				+ "{:.3f}".format(batch_dur)
				+ " sec/batch]"
			)
			# Save progress to file
			train_prog_f.write(
				str(batch_idx)
				+ " "
				+ "{:.4f}".format(class_loss.item())
				+ " "
				+ "{:.2f}".format(class_acc)
				+ " "
				+ "{:.4f}".format(conf_loss.item())
				+ " "
				+ "{:.2f}".format(avg_conf)
				+ "\n"
			)
	train_prog_f.close()


def test(args, encoder, class_out, conf_out, device, loader, signal=1.0, noise=0.0, non_blocking=False):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	hardtanh = nn.Hardtanh()
	# Iterate over batches
	all_test_correct_preds = []
	all_test_conf = []
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(loader):
			# Load data
			x = data.to(device, non_blocking=non_blocking)
			y_targ = target.to(device, non_blocking=non_blocking)
			# Scale signal
			x = x * signal
			# Scale to [-1, 1]
			x = (x - 0.5) / 0.5
			# Add noise
			x = x + torch.randn_like(x) * noise
			# Threshold image
			x = hardtanh(x)
			# Get model predictions
			z = encoder(x, device)
			y_pred = class_out(z)
			conf = conf_out(z)
			# Collect responses
			# Correct predictions
			_, y_pred_argmax = y_pred.max(1)
			correct_preds = torch.eq(y_pred_argmax, y_targ).type(torch.float)
			all_test_correct_preds.append(correct_preds.detach().cpu().numpy())
			# Confidence
			all_test_conf.append(conf.squeeze().detach().cpu().numpy())
	# Average test accuracy and confidence
	all_test_correct_preds = np.concatenate(all_test_correct_preds)
	all_test_conf = np.concatenate(all_test_conf)
	avg_test_acc = np.mean(all_test_correct_preds) * 100.0
	avg_test_conf = np.mean(all_test_conf) * 100.0
	avg_test_conf_correct = np.mean(all_test_conf[all_test_correct_preds == 1]) * 100.0
	avg_test_conf_incorrect = np.mean(all_test_conf[all_test_correct_preds == 0]) * 100.0
	# Report
	log.info(
		"[Signal = "
		+ "{:.2f}".format(signal)
		+ "] "
		+ "[Noise = "
		+ "{:.2f}".format(noise)
		+ "] "
		+ "[Class. Acc. = "
		+ "{:.2f}".format(avg_test_acc)
		+ "] "
		+ "[Conf. = "
		+ "{:.2f}".format(avg_test_conf)
		+ "] "
		+ "[Conf. (correct) = "
		+ "{:.2f}".format(avg_test_conf_correct)
		+ "] "
		+ "[Conf. (incorrect) = "
		+ "{:.2f}".format(avg_test_conf_incorrect)
		+ "]"
	)
	return avg_test_acc, avg_test_conf, avg_test_conf_correct, avg_test_conf_incorrect


def main():
	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-batch-size", type=int, default=32)
	parser.add_argument("--test-batch-size", type=int, default=100)
	parser.add_argument("--signal_range", type=list, default=[0.1, 1.0])
	parser.add_argument("--signal_range_test", type=list, default=[0.0, 1.0])
	parser.add_argument("--signal_N_test", type=int, default=100)  # 500 # 100
	parser.add_argument("--noise_range", type=list, default=[1.0, 2.0])
	parser.add_argument("--noise_N_test", type=int, default=2)
	parser.add_argument("--img_size", type=int, default=32)
	parser.add_argument("--latent_dim", type=int, default=100)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--no-cuda", action="store_true", default=False)
	parser.add_argument("--log_interval", type=int, default=10)
	parser.add_argument("--device", type=int, default=0)
	parser.add_argument("--run", type=str, default="1")
	parser.add_argument("--n_classes", type=int, default=10)
	args = parser.parse_args()

	# Device: CUDA, else MPS (if allowed), else CPU — same --no-cuda flag as v5-style (disables both CUDA and MPS)
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	use_mps = not args.no_cuda and torch.backends.mps.is_available()
	if use_cuda:
		device = torch.device("cuda:" + str(args.device))
	elif use_mps:
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	# CUDA: multiple workers + pin_memory (throughput). MPS/CPU: num_workers=0 — matches
	# train_and_eval_3.py (no worker kwargs) and avoids intermittent multiprocessing issues
	# on macOS (worker crashes / hangs with MPS are common with persistent_workers).
	loader_kwargs = {}
	if use_cuda:
		nw = min(4, os.cpu_count() or 1)
		loader_kwargs = {
			"num_workers": nw,
			"pin_memory": True,
			"persistent_workers": nw > 0,
		}

	non_blocking = bool(use_cuda and loader_kwargs.get("pin_memory"))

	# Create training/test sets from MNIST
	log.info("Loading MNIST training and test sets...")
	# Set up directory to download datasets to
	dset_dir = "./datasets"
	check_path(dset_dir)
	# Transforms
	transforms_to_apply = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor()])

	n_classes = args.n_classes
	all_digit_classes = np.arange(10)  # create a vector of int (0; 9)
	np.random.shuffle(all_digit_classes)  # random shuffle
	selected_train_classes = all_digit_classes[:n_classes]  # choose first n classes of
	log.info("Selected classes:")  # print this text
	log.info(str(selected_train_classes))  # print the value of selected classes (digits)

	train_dataset = datasets.MNIST(
		dset_dir,
		train=True,
		download=True,
		transform=transforms_to_apply,
	)  # this part will go to training
	test_dataset = datasets.MNIST(
		dset_dir,
		train=False,
		transform=transforms_to_apply,
	)  # this part will go to eval

	train_targets = _targets_as_numpy(train_dataset)
	test_targets = _targets_as_numpy(test_dataset)
	train_indices = np.where(np.isin(train_targets, selected_train_classes))[0]
	test_indices = np.where(np.isin(test_targets, selected_train_classes))[0]

	# Training set loader
	train_loader = torch.utils.data.DataLoader(
		IndexedSubset(train_dataset, train_indices),
		batch_size=args.train_batch_size,
		shuffle=True,
		drop_last=True,  # BatchNorm requires batch size > 1 in training mode
		**loader_kwargs,
	)
	# Test set loader
	test_loader = torch.utils.data.DataLoader(
		IndexedSubset(test_dataset, test_indices),
		batch_size=args.test_batch_size,
		shuffle=True,
		**loader_kwargs,
	)

	# Build model
	log.info("Building model...")
	encoder = Encoder(args).to(device)
	class_out = Class_out(args).to(device)
	conf_out = Conf_out(args).to(device)
	all_modules = nn.ModuleList([encoder, class_out, conf_out])

	# Create optimizer
	log.info("Setting up optimizer...")
	optimizer = optim.Adam(all_modules.parameters(), lr=args.lr)

	# Train
	log.info("Training begins...")
	for epoch in range(1, args.epochs + 1):
		# Training loop
		train(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch, non_blocking)

	# Test
	log.info("Test...")
	# Evaluate without noise
	test_acc_noiseless, _, __, ___ = test(
		args, encoder, class_out, conf_out, device, test_loader, signal=1, noise=0, non_blocking=non_blocking
	)
	# Signal and noise values for test
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	noise_test_vals = np.linspace(args.noise_range[0], args.noise_range[1], args.noise_N_test)
	all_test_acc = []
	all_test_conf = []
	all_test_conf_correct = []
	all_test_conf_incorrect = []
	for n in range(noise_test_vals.shape[0]):
		all_signal_test_acc = []
		all_signal_test_conf = []
		all_signal_test_conf_correct = []
		all_signal_test_conf_incorrect = []
		for s in range(signal_test_vals.shape[0]):
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = test(
				args,
				encoder,
				class_out,
				conf_out,
				device,
				test_loader,
				signal=signal_test_vals[s],
				noise=noise_test_vals[n],
				non_blocking=non_blocking,
			)
			all_signal_test_acc.append(test_acc)
			all_signal_test_conf.append(test_conf)
			all_signal_test_conf_correct.append(test_conf_correct)
			all_signal_test_conf_incorrect.append(test_conf_incorrect)
		all_test_acc.append(all_signal_test_acc)
		all_test_conf.append(all_signal_test_conf)
		all_test_conf_correct.append(all_signal_test_conf_correct)
		all_test_conf_incorrect.append(all_signal_test_conf_incorrect)
	# Save test results
	test_dir = "./test_fast/"
	check_path(test_dir)
	classes_dir = test_dir + "classes" + str(n_classes) + "/"
	check_path(classes_dir)
	model_dir = classes_dir + "run" + str(args.run) + "/"
	check_path(model_dir)
	np.savez(
		model_dir + "test_results.npz",
		signal_test_vals=signal_test_vals,
		noise_test_vals=noise_test_vals,
		test_acc_noiseless=test_acc_noiseless,
		all_test_acc=np.array(all_test_acc),
		all_test_conf=np.array(all_test_conf),
		all_test_conf_correct=np.array(all_test_conf_correct),
		all_test_conf_incorrect=np.array(all_test_conf_incorrect),
	)


if __name__ == "__main__":
	main()
