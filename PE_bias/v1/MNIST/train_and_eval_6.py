import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import argparse
import os
import sys
import time
from pathlib import Path

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from architecture import *
from util import log

class OptimizedRemappedDataset(torch.utils.data.Dataset):
	"""Optimized dataset with pre-computed indices and labels"""
	def __init__(self, dataset, indices, label_mapping):
		self.dataset = dataset
		self.indices = np.array(indices)  # Convert to numpy for faster indexing
		# Pre-compute all remapped labels - handle tensor targets
		targets = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else dataset.targets
		self.remapped_labels = np.array([label_mapping[targets[i]] for i in indices])
	
	def __len__(self):
		return len(self.indices)
	
	def __getitem__(self, idx):
		original_idx = self.indices[idx]
		image, _ = self.dataset[original_idx]  # Don't need original label
		return image, self.remapped_labels[idx]

def check_path(path):
	"""Create directory if it doesn't exist"""
	Path(path).mkdir(parents=True, exist_ok=True)

def train_optimized(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch):
	"""Optimized training function with minimal I/O and better batching"""
	# Set to training mode
	encoder.train()
	class_out.train()
	conf_out.train()
	
	# Pre-allocate tensors for batch processing
	batch_size = args.train_batch_size
	class_criterion = nn.CrossEntropyLoss()
	conf_criterion = nn.BCELoss()  # Use BCELoss like V5
	
	# Training metrics
	total_class_loss = 0.0
	total_conf_loss = 0.0
	total_class_acc = 0.0
	total_conf = 0.0
	num_batches = 0
	
	# Use tqdm for progress bar if available
	try:
		from tqdm import tqdm
		loader = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
	except ImportError:
		loader = train_loader
	
	for batch_idx, (data, target) in enumerate(loader):
		data, target = data.to(device), target.to(device)
		
		# Apply same preprocessing as V5 (optimized)
		# Scale signal
		signal = ((torch.rand(data.shape[0], device=device) * (args.signal_range[1] - args.signal_range[0])) + args.signal_range[0])
		data = data * signal.view(-1, 1, 1, 1)
		# Scale to [-1, 1]
		data = (data - 0.5) / 0.5
		# Add noise (optimized)
		noise_scale = (torch.rand(data.shape[0], device=device) * (args.noise_range[1] - args.noise_range[0])) + args.noise_range[0]
		noise = torch.randn_like(data) * noise_scale.view(-1, 1, 1, 1)
		data = data + noise
		# Threshold image
		data = nn.Hardtanh()(data)
		
		# Zero gradients
		optimizer.zero_grad()
		
		# Forward pass
		z = encoder(data, device)
		class_pred = class_out(z)
		conf_pred = conf_out(z)
		
		# Compute losses (same as V5)
		class_loss = class_criterion(class_pred, target)
		# Confidence loss: predict correctness (binary)
		correct_preds = (class_pred.argmax(dim=1) == target).type(torch.float)
		conf_loss = conf_criterion(conf_pred.squeeze(), correct_preds)
		
		# Backward pass
		loss = class_loss + conf_loss
		loss.backward()
		optimizer.step()
		
		# Compute metrics
		with torch.no_grad():
			class_acc = (class_pred.argmax(dim=1) == target).float().mean().item()
			conf_mean = conf_pred.mean().item()
			
			total_class_loss += class_loss.item()
			total_conf_loss += conf_loss.item()
			total_class_acc += class_acc
			total_conf += conf_mean
			num_batches += 1
		
		# Log every 10 batches instead of every batch
		if batch_idx % 10 == 0:
			log.info(f'[Epoch: {epoch}] [Batch: {batch_idx} of {len(train_loader)}] '
					f'[Class. Loss = {class_loss.item():.4f}] [Class. Acc. = {class_acc*100:.2f}] '
					f'[Conf. Loss = {conf_loss.item():.4f}] [Conf. = {conf_mean*100:.2f}]')
	
	# Log epoch summary
	avg_class_loss = total_class_loss / num_batches
	avg_conf_loss = total_conf_loss / num_batches
	avg_class_acc = total_class_acc / num_batches
	avg_conf = total_conf / num_batches
	
	log.info(f'Epoch {epoch} Summary: Class Loss = {avg_class_loss:.4f}, '
			f'Class Acc = {avg_class_acc*100:.2f}%, Conf Loss = {avg_conf_loss:.4f}, '
			f'Conf = {avg_conf*100:.2f}%')

def test_optimized(args, encoder, class_out, conf_out, device, test_loader, signal=1, noise=0):
	"""Optimized test function with vectorized operations"""
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	
	all_class_acc = []
	all_conf = []
	all_conf_correct = []
	all_conf_incorrect = []
	
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			
			# Apply same preprocessing as V5 (optimized)
			# Scale signal
			data = data * signal
			# Scale to [-1, 1]
			data = (data - 0.5) / 0.5
			# Add noise (optimized)
			data = data + torch.randn_like(data) * noise
			# Threshold image
			data = nn.Hardtanh()(data)
			
			# Forward pass
			z = encoder(data, device)
			class_pred = class_out(z)
			conf_pred = conf_out(z)
			
			# Compute accuracy
			class_acc = (class_pred.argmax(dim=1) == target).float()
			all_class_acc.append(class_acc)
			
			# Compute confidence
			conf = conf_pred.squeeze()
			all_conf.append(conf)
			
			# Separate confidence for correct/incorrect predictions
			correct_mask = class_acc.bool()
			all_conf_correct.append(conf[correct_mask])
			all_conf_incorrect.append(conf[~correct_mask])
	
	# Concatenate all results
	all_class_acc = torch.cat(all_class_acc)
	all_conf = torch.cat(all_conf)
	all_conf_correct = torch.cat(all_conf_correct) if all_conf_correct[0].numel() > 0 else torch.tensor([])
	all_conf_incorrect = torch.cat(all_conf_incorrect) if all_conf_incorrect[0].numel() > 0 else torch.tensor([])
	
	# Compute final metrics
	test_acc = all_class_acc.mean().item() * 100
	test_conf = all_conf.mean().item() * 100
	test_conf_correct = all_conf_correct.mean().item() * 100 if all_conf_correct.numel() > 0 else 0
	test_conf_incorrect = all_conf_incorrect.mean().item() * 100 if all_conf_incorrect.numel() > 0 else 0
	
	return test_acc, test_conf, test_conf_correct, test_conf_incorrect

def create_optimized_dataloaders(args, selected_train_classes, label_mapping):
	"""Create optimized data loaders with better memory management"""
	# Use same transforms as V5
	transforms_to_apply = transforms.Compose([
		transforms.Resize(args.img_size), 
		transforms.ToTensor()
	])
	
	# Load datasets
	train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transforms_to_apply)
	test_dataset = datasets.MNIST('./datasets', train=False, transform=transforms_to_apply)
	
	# Find indices more efficiently
	train_targets = train_dataset.targets.numpy()
	test_targets = test_dataset.targets.numpy()
	
	train_indices = np.where(np.isin(train_targets, selected_train_classes))[0]
	test_indices = np.where(np.isin(test_targets, selected_train_classes))[0]
	
	# Create optimized datasets
	train_dataset_opt = OptimizedRemappedDataset(train_dataset, train_indices, label_mapping)
	test_dataset_opt = OptimizedRemappedDataset(test_dataset, test_indices, label_mapping)
	
	# Create data loaders with optimized settings
	kwargs = {'num_workers': min(4, os.cpu_count()), 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': min(4, os.cpu_count())}
	
	train_loader = torch.utils.data.DataLoader(
		train_dataset_opt, batch_size=args.train_batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		test_dataset_opt, batch_size=args.test_batch_size, shuffle=False, **kwargs)
	
	return train_loader, test_loader

def main():
	# Settings
	parser = argparse.ArgumentParser(description='Optimized MNIST Training and Evaluation')
	parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training')
	parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing')
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
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')
	log.info(f'Using device: {device}')
	
	# Set random seeds for reproducibility
	torch.manual_seed(42)
	np.random.seed(42)
	
	# Phase 1: Train with n_classes
	n_classes = args.n_classes
	log.info(f'Phase 1: Training with {n_classes} classes')
	
	# Select classes
	all_digit_classes = np.arange(10)
	np.random.shuffle(all_digit_classes)
	selected_train_classes = all_digit_classes[:n_classes]
	log.info(f'Selected classes: {selected_train_classes}')
	
	# Create label mapping
	label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_train_classes)}
	
	# Create optimized data loaders
	train_loader, test_loader = create_optimized_dataloaders(args, selected_train_classes, label_mapping)
	
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
		train_optimized(args, encoder, class_out_10, conf_out_10, device, train_loader, optimizer, epoch)
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
	train_loader, test_loader = create_optimized_dataloaders(args, selected_train_classes, label_mapping)
	
	# Freeze encoder and create new heads
	log.info('Building final model...')
	for param in encoder.parameters():
		param.requires_grad = False
	
	class_out = Class_out(args, output_dim=new_n_classes).to(device)
	conf_out = Conf_out(args).to(device)
	all_modules = nn.ModuleList([encoder, class_out, conf_out])
	
	# Create optimizer for Phase 2
	log.info('Setting up final optimizer...')
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, all_modules.parameters()), lr=args.lr)
	
	# Train Phase 2
	log.info('Phase 2 training begins...')
	start_time = time.time()
	for epoch in range(1, args.epochs + 1):
		train_optimized(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch)
	phase2_time = time.time() - start_time
	log.info(f'Phase 2 completed in {phase2_time:.2f} seconds')
	
	# Optimized testing
	log.info('Testing...')
	start_time = time.time()
	
	# Evaluate without noise
	test_acc_noiseless, _, __, ___ = test_optimized(args, encoder, class_out, conf_out, device, test_loader, signal=1, noise=0)
	
	# Create test parameter grids
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	noise_test_vals = np.linspace(args.noise_range[0], args.noise_range[1], args.noise_N_test)
	
	# Vectorized testing - test all combinations efficiently
	all_test_acc = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	all_test_conf = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	all_test_conf_correct = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	all_test_conf_incorrect = np.zeros((len(noise_test_vals), len(signal_test_vals)))
	
	for n_idx, noise_val in enumerate(noise_test_vals):
		for s_idx, signal_val in enumerate(signal_test_vals):
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = test_optimized(
				args, encoder, class_out, conf_out, device, test_loader, 
				signal=signal_val, noise=noise_val)
			
			all_test_acc[n_idx, s_idx] = test_acc
			all_test_conf[n_idx, s_idx] = test_conf
			all_test_conf_correct[n_idx, s_idx] = test_conf_correct
			all_test_conf_incorrect[n_idx, s_idx] = test_conf_incorrect
			
			# Log progress for long test runs
			if (n_idx * len(signal_test_vals) + s_idx) % 5 == 0 or (n_idx * len(signal_test_vals) + s_idx) < 10:
				log.info(f'[Signal = {signal_val:.2f}] [Noise = {noise_val:.2f}] '
						f'[Class. Acc. = {test_acc:.2f}] [Conf. = {test_conf:.2f}]')
	
	test_time = time.time() - start_time
	log.info(f'Testing completed in {test_time:.2f} seconds')
	
	# Save results
	test_dir = './test/'
	check_path(test_dir)
	classes_dir = test_dir + f'classes{str(n_classes)}_{str(new_n_classes)}_v6/'
	check_path(classes_dir)
	model_dir = classes_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	
	np.savez(model_dir + 'test_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=noise_test_vals,
			 test_acc_noiseless=test_acc_noiseless,
			 all_test_acc=all_test_acc,
			 all_test_conf=all_test_conf,
			 all_test_conf_correct=all_test_conf_correct,
			 all_test_conf_incorrect=all_test_conf_incorrect)
	
	total_time = phase1_time + phase2_time + test_time
	log.info(f'Total execution time: {total_time:.2f} seconds')
	log.info(f'Results saved to: {model_dir}')

if __name__ == '__main__':
	main()
