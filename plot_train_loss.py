import re
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def parse_training_log(log_file_path):
    """Parse training log file to extract loss and epoch information"""
    losses = []
    epochs = []
    learning_rates = []
    grad_norms = []
    
    if not os.path.exists(log_file_path):
        print(f"Error: File {log_file_path} does not exist")
        return None, None, None, None
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Match lines containing training metrics, format like: {'loss': 0.3762, 'grad_norm': 0.32876765727996826, 'learning_rate': 0.00014625, 'epoch': 0.82}
                match = re.search(r"\{'loss': ([\d\.]+),\s*'grad_norm': ([\d\.e\-\+]+),\s*'learning_rate': ([\d\.e\-\+]+),\s*'epoch': ([\d\.]+)\}", line)
                if match:
                    loss = float(match.group(1))
                    grad_norm = float(match.group(2))
                    learning_rate = float(match.group(3))
                    epoch = float(match.group(4))
                    
                    losses.append(loss)
                    epochs.append(epoch)
                    learning_rates.append(learning_rate)
                    grad_norms.append(grad_norm)
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None, None
    
    if not losses:
        print("No valid training data found")
        return None, None, None, None
    
    print(f"Successfully parsed {len(losses)} training data points")
    return losses, epochs, learning_rates, grad_norms

def plot_training_curves(losses, epochs, learning_rates, grad_norms, save_path=None):
    """Plot training curves"""
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Loss curve
    ax1.plot(epochs, losses, 'b-', marker='o', markersize=3, linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot learning rate curve
    ax2.plot(epochs, learning_rates, 'r-', marker='s', markersize=3, linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot gradient norm curve
    ax3.plot(epochs, grad_norms, 'g-', marker='^', markersize=3, linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norm Curve')
    ax3.grid(True, alpha=0.3)
    
    # Plot moving average of Loss
    if len(losses) > 5:
        window_size = max(5, len(losses) // 20)  # Dynamic window size
        moving_avg = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(losses), i + window_size // 2 + 1)
            moving_avg.append(sum(losses[start_idx:end_idx]) / (end_idx - start_idx))
        
        ax4.plot(epochs, losses, 'b-', alpha=0.3, linewidth=1, label='Original Loss')
        ax4.plot(epochs, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Loss with Moving Average')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(bottom=0)
    else:
        ax4.plot(epochs, losses, 'b-', marker='o', markersize=3, linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    plt.show()

def print_training_summary(losses, epochs, learning_rates):
    """Print training summary information"""
    if not losses:
        return
    
    print("\n=== Training Summary ===")
    print(f"Total training epochs: {epochs[-1]:.2f} epochs")
    print(f"Total training steps: {len(losses)} steps")
    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Min Loss: {min(losses):.4f}")
    print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    print(f"Initial learning rate: {learning_rates[0]:.2e}")
    print(f"Final learning rate: {learning_rates[-1]:.2e}")

def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves')
    parser.add_argument('--log_file', '-f', type=str, default='finetune_llm_domain.log',
                        help='Training log file path (default: finetune_llm_domain.log)')
    parser.add_argument('--save', '-s', type=str, 
                        help='Path to save the image (optional)')
    parser.add_argument('--auto_save', '-a', action='store_true',
                        help='Automatically save image to plots directory')
    
    args = parser.parse_args()
    
    # Parse log file
    losses, epochs, learning_rates, grad_norms = parse_training_log(args.log_file)
    
    if losses is None:
        return
    
    # Print training summary
    print_training_summary(losses, epochs, learning_rates)
    
    # Determine save path
    save_path = args.save
    if args.auto_save:
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'plots/training_curves_{timestamp}.png'
    
    # Plot curves
    plot_training_curves(losses, epochs, learning_rates, grad_norms, save_path)

if __name__ == "__main__":
    main() 