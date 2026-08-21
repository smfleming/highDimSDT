#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse
import logging
import time
import os
from pathlib import Path
from architecture import Encoder, Class_out, Conf_out

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

def check_path(path):
	"""Create directory if it doesn't exist"""
	Path(path).mkdir(parents=True, exist_ok=True)

class UltraOptimizedDataset(torch.utils.data.Dataset):
	"""Ultra-optimized dataset with pre-computed everything"""
	def __init__(self, dataset, indices, label_mapping):
		self.dataset = dataset
		self.indices = np.array(indices, dtype=np.int32)  # Use int32 for speed
		# Pre-compute all remapped labels
		targets = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else dataset.targets
		self.remapped_labels = np.array([label_mapping[targets[i]] for i in indices], dtype=np.int64)
	
	def __len__(self):
		return len(self.indices)
	
	def __getitem__(self, idx):
		# Direct indexing without bounds checking for speed
		original_idx = self.indices[idx]
		image, _ = self.dataset[original_idx]  # Don't need original label
		return image, self.remapped_labels[idx]

def train_ultra_optimized(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch):
	"""Ultra-optimized training with maximum performance"""
	encoder.train()
	class_out.train()
	conf_out.train()
	
	# Pre-compute constants
	signal_range_diff = args.signal_range[1] - args.signal_range[0]
	noise_range_diff = args.noise_range[1] - args.noise_range[0]
	
	# Loss functions
	class_criterion = nn.CrossEntropyLoss()
	conf_criterion = nn.BCELoss()
	
	# Pre-allocate tensors for metrics (avoid Python loops)
	num_batches = len(train_loader)
	metrics = torch.zeros(4, num_batches, device=device)  # [class_loss, conf_loss, class_acc, conf_mean]
	
	# Ultra-fast training loop
	for batch_idx, (data, target) in enumerate(train_loader):
		# Non-blocking transfer for maximum speed
		data = data.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)
		
		# Ultra-optimized preprocessing (all GPU operations)
		# Signal scaling (single operation)
		signal = torch.rand(data.shape[0], device=device) * signal_range_diff + args.signal_range[0]
		data = data * signal.view(-1, 1, 1, 1)
		
		# Normalization (in-place)
		data.sub_(0.5).div_(0.5)
		
		# Noise (single operation)
		noise_scale = torch.rand(data.shape[0], device=device) * noise_range_diff + args.noise_range[0]
		data.add_(torch.randn_like(data) * noise_scale.view(-1, 1, 1, 1))
		
		# Thresholding (in-place, faster than Hardtanh)
		data.clamp_(-1.0, 1.0)
		
		# Forward pass
		optimizer.zero_grad()
		z = encoder(data, device)
		class_pred = class_out(z)
		conf_pred = conf_out(z)
		
		# Loss computation
		class_loss = class_criterion(class_pred, target)
		correct_preds = (class_pred.argmax(dim=1) == target).float()
		conf_loss = conf_criterion(conf_pred.squeeze(), correct_preds)
		
		# Backward pass
		loss = class_loss + conf_loss
		loss.backward()
		optimizer.step()
		
		# Store metrics (vectorized)
		with torch.no_grad():
			metrics[0, batch_idx] = class_loss
			metrics[1, batch_idx] = conf_loss
			metrics[2, batch_idx] = correct_preds.mean()
			metrics[3, batch_idx] = conf_pred.mean()
		
		# Minimal logging
		if batch_idx % 50 == 0:
			log.info(f'[Epoch: {epoch}] [Batch: {batch_idx}/{num_batches}] '
					f'[Class. Loss = {class_loss.item():.4f}] [Class. Acc. = {correct_preds.mean().item()*100:.2f}] '
					f'[Conf. Loss = {conf_loss.item():.4f}] [Conf. = {conf_pred.mean().item()*100:.2f}]')
	
	# Vectorized summary
	avg_metrics = metrics.mean(dim=1)
	log.info(f'Epoch {epoch} Summary: Class Loss = {avg_metrics[0]:.4f}, Class Acc = {avg_metrics[2]*100:.2f}%, '
			f'Conf Loss = {avg_metrics[1]:.4f}, Conf = {avg_metrics[3]*100:.2f}%')

def test_ultra_optimized(args, encoder, class_out, conf_out, device, test_loader, signal=1, noise=0):
	"""Ultra-optimized test function"""
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	
	# Pre-allocate result tensors
	all_results = []
	
	with torch.no_grad():
		for data, target in test_loader:
			data = data.to(device, non_blocking=True)
			target = target.to(device, non_blocking=True)
			
			# Ultra-fast preprocessing
			data = data * signal
			data.sub_(0.5).div_(0.5)
			data.add_(torch.randn_like(data) * noise)
			data.clamp_(-1.0, 1.0)
			
			# Forward pass
			z = encoder(data, device)
			class_pred = class_out(z)
			conf_pred = conf_out(z)
			
			# Compute metrics
			class_acc = (class_pred.argmax(dim=1) == target).float()
			conf = conf_pred.squeeze()
			
			# Separate correct/incorrect
			correct_mask = class_acc.bool()
			conf_correct = conf[correct_mask] if correct_mask.any() else torch.tensor([], device=device)
			conf_incorrect = conf[~correct_mask] if (~correct_mask).any() else torch.tensor([], device=device)
			
			all_results.append({
				'class_acc': class_acc,
				'conf': conf,
				'conf_correct': conf_correct,
				'conf_incorrect': conf_incorrect
			})
	
	# Concatenate all results
	all_class_acc = torch.cat([r['class_acc'] for r in all_results])
	all_conf = torch.cat([r['conf'] for r in all_results])
	all_conf_correct = torch.cat([r['conf_correct'] for r in all_results if r['conf_correct'].numel() > 0])
	all_conf_incorrect = torch.cat([r['conf_incorrect'] for r in all_results if r['conf_incorrect'].numel() > 0])
	
	# Compute final metrics
	test_acc = all_class_acc.mean().item() * 100
	test_conf = all_conf.mean().item() * 100
	test_conf_correct = all_conf_correct.mean().item() * 100 if all_conf_correct.numel() > 0 else 0
	test_conf_incorrect = all_conf_incorrect.mean().item() * 100 if all_conf_incorrect.numel() > 0 else 0
	
	return test_acc, test_conf, test_conf_correct, test_conf_incorrect

def create_ultra_optimized_dataloaders(args, selected_train_classes, label_mapping):
	"""Create ultra-optimized data loaders"""
	# Same transforms as V5
	transforms_to_apply = transforms.Compose([
		transforms.Resize(args.img_size), 
		transforms.ToTensor()
	])
	
	# Load datasets
	train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transforms_to_apply)
	test_dataset = datasets.MNIST('./datasets', train=False, transform=transforms_to_apply)
	
	# Find indices efficiently
	train_targets = train_dataset.targets.numpy()
	test_targets = test_dataset.targets.numpy()
	
	train_indices = np.where(np.isin(train_targets, selected_train_classes))[0]
	test_indices = np.where(np.isin(test_targets, selected_train_classes))[0]
	
	# Create ultra-optimized datasets
	train_dataset_opt = UltraOptimizedDataset(train_dataset, train_indices, label_mapping)
	test_dataset_opt = UltraOptimizedDataset(test_dataset, test_indices, label_mapping)
	
	# Ultra-optimized data loaders
	kwargs = {
		'num_workers': min(8, os.cpu_count()),  # More workers
		'pin_memory': True,
		'persistent_workers': True,  # Keep workers alive
		'prefetch_factor': 4  # Prefetch more batches
	} if torch.cuda.is_available() else {
		'num_workers': min(8, os.cpu_count()),
		'persistent_workers': True,
		'prefetch_factor': 4
	}
	
	train_loader = DataLoader(train_dataset_opt, batch_size=args.train_batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(test_dataset_opt, batch_size=args.test_batch_size, shuffle=False, **kwargs)
	
	return train_loader, test_loader

def main():
	parser = argparse.ArgumentParser(description='Ultra-Optimized MNIST Training and Evaluation')
	parser.add_argument('--train_batch_size', type=int, default=128, help='input batch size for training')  # Larger default
	parser.add_argument('--test_batch_size', type=int, default=2000, help='input batch size for testing')  # Larger default
	parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--latent_dim', type=int, default=32, help='latent dimensionality')
	parser.add_argument('--img_size', type=int, default=32, help='image size for resizing')
	parser.add_argument('--run', type=int, default=1, help='run number')
	parser.add_argument('--n_classes', type=int, help='number of classes for first phase')
	parser.add_argument('--new_n_classes', type=int, help='number of classes for second phase')
	parser.add_argument('--signal_range', type=float, nargs=2, default=[0.1, 1.0], help='signal range for training')
	parser.add_argument('--signal_range_test', type=float, nargs=2, default=[0, 1], help='signal range for testing')
	parser.add_argument('--signal_N_test', type=int, default=100, help='number of signal values for testing')
	parser.add_argument('--noise_range', type=float, nargs=2, default=[1.0, 2.0], help='noise range for training and testing')
	parser.add_argument('--noise_N_test', type=int, default=2, help='number of noise values for testing')
	args = parser.parse_args()
	
	# Set device (CUDA, MPS, or CPU)
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')
	log.info(f'Using device: {device}')
	
	# Set random seeds for reproducibility
	torch.manual_seed(42)
	np.random.seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)
		torch.cuda.manual_seed_all(42)
	
	# Phase 1: Training with n_classes
	n_classes = args.n_classes
	log.info(f'Phase 1: Training with {n_classes} classes')
	
	# Select classes
	all_digit_classes = np.arange(10)
	np.random.shuffle(all_digit_classes)
	selected_train_classes = all_digit_classes[:n_classes]
	log.info(f'Selected classes: {selected_train_classes}')
	
	# Create label mapping
	label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_train_classes)}
	
	# Create ultra-optimized data loaders
	train_loader, test_loader = create_ultra_optimized_dataloaders(args, selected_train_classes, label_mapping)
	
	# Build model
	log.info('Building model...')
	encoder = Encoder(args).to(device)
	class_out_10 = Class_out(args, output_dim=n_classes).to(device)
	conf_out_10 = Conf_out(args).to(device)
	all_modules = nn.ModuleList([encoder, class_out_10, conf_out_10])
	
	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(all_modules.parameters(), lr=args.lr)
	
	# Train Phase 1
	log.info('Phase 1 training begins...')
	start_time = time.time()
	for epoch in range(1, args.epochs + 1):
		train_ultra_optimized(args, encoder, class_out_10, conf_out_10, device, train_loader, optimizer, epoch)
	phase1_time = time.time() - start_time
	log.info(f'Phase 1 completed in {phase1_time:.2f} seconds')
	
	# Phase 2: Transfer learning with new_n_classes
	new_n_classes = args.new_n_classes
	log.info(f'Phase 2: Transfer learning with {new_n_classes} classes')
	
	# Select new classes (reuse the same selection for consistency)
	selected_train_classes = all_digit_classes[:new_n_classes]
	log.info(f'Selected classes: {selected_train_classes}')
	
	# Create new label mapping
	label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_train_classes)}
	
	# Create new data loaders
	train_loader, test_loader = create_ultra_optimized_dataloaders(args, selected_train_classes, label_mapping)
	
	# Freeze encoder and create new heads
	old_encoder = torch.jit.script(encoder)  # JIT compile for speed
	for param in encoder.parameters():
		param.requires_grad = False
	
	class_out = Class_out(args, output_dim=new_n_classes).to(device)
	conf_out = Conf_out(args).to(device)
	
	# Create optimizer for trainable parameters only
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
								 [*class_out.parameters(), *conf_out.parameters()]), lr=args.lr)
	
	# Train Phase 2
	log.info('Phase 2 training begins...')
	start_time = time.time()
	for epoch in range(1, args.epochs + 1):
		train_ultra_optimized(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch)
	phase2_time = time.time() - start_time
	log.info(f'Phase 2 completed in {phase2_time:.2f} seconds')
	
	# Testing
	log.info('Testing...')
	start_time = time.time()
	
	# Create test parameter grids
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	noise_test_vals = np.linspace(args.noise_range[0], args.noise_range[1], args.noise_N_test)
	
	# Vectorized testing
	all_test_acc = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	all_test_conf = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	all_test_conf_correct = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	all_test_conf_incorrect = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	
	for n_idx, noise_val in enumerate(noise_test_vals):
		for s_idx, signal_val in enumerate(signal_test_vals):
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = test_ultra_optimized(
				args, encoder, class_out, conf_out, device, test_loader, 
				signal=signal_val, noise=noise_val)
			
			all_test_acc[n_idx, s_idx] = test_acc
			all_test_conf[n_idx, s_idx] = test_conf
			all_test_conf_correct[n_idx, s_idx] = test_conf_correct
			all_test_conf_incorrect[n_idx, s_idx] = test_conf_incorrect
			
			# Minimal logging
			if (n_idx * len(signal_test_vals) + s_idx) % 10 == 0:
				log.info(f'[Signal = {signal_val:.2f}] [Noise = {noise_val:.2f}] '
						f'[Class. Acc. = {test_acc:.2f}] [Conf. = {test_conf:.2f}]')
	
	test_time = time.time() - start_time
	log.info(f'Testing completed in {test_time:.2f} seconds')
	
	# Save results
	test_dir = './test/'
	check_path(test_dir)
	classes_dir = test_dir + f'classes{str(n_classes)}_{str(new_n_classes)}_v6_ultra/'
	check_path(classes_dir)
	run_dir = classes_dir + 'run' + str(args.run) + '/'
	check_path(run_dir)
	
	# Compute noiseless test accuracy
	test_acc_noiseless = all_test_acc[0, -1] if len(noise_test_vals) > 0 else 0
	
	# Save results
	np.savez(run_dir + 'test_results.npz',
			 all_test_acc=all_test_acc,
			 all_test_conf=all_test_conf,
			 all_test_conf_correct=all_test_conf_correct,
			 all_test_conf_incorrect=all_test_conf_incorrect,
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=noise_test_vals,
			 test_acc_noiseless=test_acc_noiseless)
	
	total_time = phase1_time + phase2_time + test_time
	log.info(f'Total execution time: {total_time:.2f} seconds')
	log.info(f'Results saved to: {run_dir}')

if __name__ == '__main__':
	main()



