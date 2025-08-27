import re
import matplotlib.pyplot as plt
import os
from datetime import datetime

def parse_training_log(log_file_path):
    """Parse training log file to extract loss and epoch information"""
    losses = []
    epochs = []
    learning_rates = []
    grad_norms = []
    
    if not os.path.exists(log_file_path):
        print(f"Warning: File {log_file_path} does not exist")
        return None, None, None, None
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Match lines containing training metrics
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
        print(f"Error reading file {log_file_path}: {e}")
        return None, None, None, None
    
    return losses, epochs, learning_rates, grad_norms

def compare_training_results():
    """Compare results of three training tasks"""
    
    # Define training tasks
    tasks = {
        'Domain Recognition': 'finetune_llm_domain.log',
        'State Extraction': 'finetune_llm_state.log', 
        'Response Generation': 'finetune_llm_response.log'
    }
    
    # Parse all log files
    results = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    for i, (task_name, log_file) in enumerate(tasks.items()):
        losses, epochs, learning_rates, grad_norms = parse_training_log(log_file)
        if losses is not None:
            results[task_name] = {
                'losses': losses,
                'epochs': epochs,
                'learning_rates': learning_rates,
                'grad_norms': grad_norms,
                'color': colors[i]
            }
            print(f"✅ {task_name}: Parsed {len(losses)} data points")
        else:
            print(f"❌ {task_name}: Parsing failed")
    
    if not results:
        print("No valid training data found")
        return
    
    # Create comprehensive comparison chart with larger fonts
    plt.rcParams.update({'font.size': 14})  # Set global font size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Loss curve comparison
    for task_name, data in results.items():
        ax1.plot(data['epochs'], data['losses'], 
                color=data['color'], marker='o', markersize=2, 
                linewidth=1.5, alpha=0.8, label=task_name)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.set_title('Training Loss Comparison', fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # 2. Learning rate curve comparison
    for task_name, data in results.items():
        ax2.plot(data['epochs'], data['learning_rates'],
                color=data['color'], marker='s', markersize=2,
                linewidth=1.5, alpha=0.8, label=task_name)
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.set_ylabel('Learning Rate', fontsize=16)
    ax2.set_title('Learning Rate Schedule Comparison', fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # 3. Gradient norm comparison
    for task_name, data in results.items():
        ax3.plot(data['epochs'], data['grad_norms'],
                color=data['color'], marker='^', markersize=2,
                linewidth=1.5, alpha=0.8, label=task_name)
    ax3.set_xlabel('Epoch', fontsize=16)
    ax3.set_ylabel('Gradient Norm', fontsize=16)
    ax3.set_title('Gradient Norm Comparison', fontsize=18)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    # 4. Loss reduction rate and final performance comparison (bar chart)
    task_names = list(results.keys())
    initial_losses = [results[task]['losses'][0] for task in task_names]
    final_losses = [results[task]['losses'][-1] for task in task_names]
    min_losses = [min(results[task]['losses']) for task in task_names]
    loss_reduction = [((initial - final) / initial * 100) for initial, final in zip(initial_losses, final_losses)]
    
    x = range(len(task_names))
    width = 0.25
    
    bars1 = ax4.bar([i - width for i in x], initial_losses, width, 
                   label='Initial Loss', alpha=0.7, color='lightcoral')
    bars2 = ax4.bar(x, final_losses, width, 
                   label='Final Loss', alpha=0.7, color='lightblue')  
    bars3 = ax4.bar([i + width for i in x], min_losses, width,
                   label='Min Loss', alpha=0.7, color='lightgreen')
    
    ax4.set_xlabel('Training Tasks', fontsize=16)
    ax4.set_ylabel('Loss', fontsize=16)
    ax4.set_title('Loss Performance Comparison', fontsize=18)
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.replace(' ', '\n') for name in task_names], fontsize=12)
    ax4.legend(fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='y', which='major', labelsize=14)
    
    # Add loss reduction percentage annotations on bar chart
    for i, (reduction, final_loss) in enumerate(zip(loss_reduction, final_losses)):
        ax4.text(i, final_loss + 0.05, f'-{reduction:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Save comparison chart
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'plots/training_comparison_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive comparison chart saved to: {save_path}")
    
    plt.show()
    
    # Print detailed comparison results
    print("\n" + "="*60)
    print("🎯 Performance Comparison of Three Training Tasks")
    print("="*60)
    
    for task_name, data in results.items():
        losses = data['losses']
        epochs = data['epochs'] 
        learning_rates = data['learning_rates']
        
        print(f"\n📋 {task_name}:")
        print(f"   • Training epochs: {epochs[-1]:.2f} epochs ({len(losses)} steps)")
        print(f"   • Initial Loss: {losses[0]:.4f}")
        print(f"   • Final Loss: {losses[-1]:.4f}")
        print(f"   • Min Loss: {min(losses):.4f}")
        print(f"   • Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
        print(f"   • Learning rate: {learning_rates[0]:.2e} → {learning_rates[-1]:.2e}")
    
    # Ranking analysis
    print(f"\n🏆 Performance Ranking:")
    print(f"   📉 Highest loss reduction rate: {max(results.keys(), key=lambda x: (results[x]['losses'][0] - results[x]['losses'][-1]) / results[x]['losses'][0] * 100)}")
    print(f"   🎯 Lowest final loss: {min(results.keys(), key=lambda x: results[x]['losses'][-1])}")
    print(f"   🏁 Lowest loss record: {min(results.keys(), key=lambda x: min(results[x]['losses']))}")
    
    return results

if __name__ == "__main__":
    compare_training_results() 