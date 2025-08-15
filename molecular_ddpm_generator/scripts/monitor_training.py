# scripts/monitor_training.py - Monitor DDPM training progress
import os
import sys
import time
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime

def parse_log_file(log_path):
    """Parse training log file"""
    if not os.path.exists(log_path):
        return None
    
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except:
                continue
    
    return logs

def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_percent = (memory_used / memory_total) * 100
        
        print(f"🚀 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB ({usage_percent:.1f}%)")
        return memory_used, memory_total, usage_percent
    else:
        print("💻 No GPU available")
        return 0, 0, 0

def check_model_files(model_dir):
    """Check saved model files"""
    if not os.path.exists(model_dir):
        return []
    
    model_files = list(Path(model_dir).glob("*.pth"))
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return model_files

def plot_training_curves(logs, save_path=None):
    """Plot training curves"""
    if not logs:
        return
    
    epochs = [log.get('epoch', 0) for log in logs if 'epoch' in log]
    train_losses = [log.get('train_total_loss', 0) for log in logs if 'train_total_loss' in log]
    val_losses = [log.get('val_total_loss', 0) for log in logs if 'val_total_loss' in log]
    
    if not epochs:
        return
    
    plt.figure(figsize=(12, 4))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss', color='blue')
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate
    plt.subplot(1, 2, 2)
    learning_rates = [log.get('learning_rate', 0) for log in logs if 'learning_rate' in log]
    if learning_rates:
        plt.plot(epochs[:len(learning_rates)], learning_rates, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training curves saved to: {save_path}")
    else:
        plt.show()

def print_training_status(model_dir, log_path=None):
    """Print current training status"""
    print("📊 DDPM Training Status")
    print("=" * 50)
    print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU status
    monitor_gpu_usage()
    
    # Model files
    model_files = check_model_files(model_dir)
    if model_files:
        print(f"\n💾 Model files ({len(model_files)}):")
        for i, model_file in enumerate(model_files[:5]):  # Show latest 5
            size_mb = model_file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            print(f"   {i+1}. {model_file.name}")
            print(f"      Size: {size_mb:.1f}MB, Modified: {mtime.strftime('%H:%M:%S')}")
        
        # Load latest model info
        try:
            latest_model = model_files[0]
            checkpoint = torch.load(latest_model, map_location='cpu')
            print(f"\n📈 Latest model ({latest_model.name}):")
            print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"   Loss: {checkpoint.get('loss', 'unknown')}")
            print(f"   Best Val Loss: {checkpoint.get('best_val_loss', 'unknown')}")
        except Exception as e:
            print(f"   ⚠️  Could not load model info: {e}")
    else:
        print("\n💾 No model files found")
    
    # Parse logs if available
    if log_path and os.path.exists(log_path):
        logs = parse_log_file(log_path)
        if logs:
            latest_log = logs[-1]
            print(f"\nLatest log entry:")
            for key, value in latest_log.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")

def continuous_monitor(model_dir, log_path=None, refresh_interval=30):
    """Continuously monitor training"""
    print("Starting continuous monitoring...")
    print(f"Model directory: {model_dir}")
    print(f"Refresh interval: {refresh_interval}s")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
            print_training_status(model_dir, log_path)
            print(f"\n⏰ Next update in {refresh_interval}s...")
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    parser = argparse.ArgumentParser(description='Monitor DDPM Training')
    parser.add_argument('--model_dir', type=str, default='models/ddpm',
                       help='Directory containing model checkpoints')
    parser.add_argument('--log_path', type=str, 
                       help='Path to training log file')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous monitoring mode')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval for continuous mode (seconds)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate training curves plot')
    parser.add_argument('--save_plot', type=str,
                       help='Save plot to file')
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitor(args.model_dir, args.log_path, args.refresh)
    else:
        print_training_status(args.model_dir, args.log_path)
        
        # Generate plots if requested
        if args.plot and args.log_path:
            logs = parse_log_file(args.log_path)
            plot_training_curves(logs, args.save_plot)

if __name__ == "__main__":
    main()