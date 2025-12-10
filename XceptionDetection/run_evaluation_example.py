"""
Quick Example - How to Run the Evaluation

This is a simple example showing how to run the evaluation.
Just update the paths below and run this script!
"""

import subprocess
import os

# =============================================================================
# UPDATE THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Path to your validation dataset (containing 0_real and 1_fake folders)
VALIDATION_DATA = "./dataset/val/"

# Path to your trained model checkpoint
CHECKPOINT = "./checkpoints/xception_experiment/latest_net.pth"

# Where to save results
OUTPUT_DIR = "./evaluation_results"

# Optional: Adjust these if needed
BATCH_SIZE = 32
THRESHOLD = 0.5
GPU_ID = 0  # Use -1 for CPU

# =============================================================================
# RUN EVALUATION
# =============================================================================

def run_evaluation():
    """Run the evaluation script"""
    
    cmd = [
        "python", "evaluate_model.py",
        "--dataroot", VALIDATION_DATA,
        "--checkpoint", CHECKPOINT,
        "--output_dir", OUTPUT_DIR,
        "--batch_size", str(BATCH_SIZE),
        "--threshold", str(THRESHOLD),
        "--gpu_id", str(GPU_ID)
    ]
    
    print("="*70)
    print("RUNNING MODEL EVALUATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Validation Data: {VALIDATION_DATA}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  GPU ID: {GPU_ID}")
    print(f"\nCommand: {' '.join(cmd)}")
    print("\n" + "="*70 + "\n")
    
    # Check if paths exist
    if not os.path.exists(VALIDATION_DATA):
        print(f"❌ ERROR: Validation data not found at: {VALIDATION_DATA}")
        print("Please update VALIDATION_DATA path in this script.")
        return False
    
    if not os.path.exists(CHECKPOINT):
        print(f"❌ ERROR: Checkpoint not found at: {CHECKPOINT}")
        print("Please update CHECKPOINT path in this script.")
        return False
    
    # Run evaluation
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running evaluation: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Error: evaluate_model.py not found.")
        print("Make sure you're in the XceptionDetection directory.")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XCEPTION MODEL EVALUATION - QUICK START")
    print("="*70)
    print("\n📋 This script will:")
    print("  1. Load your trained model")
    print("  2. Run inference on validation data")
    print("  3. Calculate 12+ performance metrics")
    print("  4. Generate 7 publication-quality visualizations")
    print("\n⚠️  Before running, make sure to update the paths at the top of this script!")
    print("\n" + "="*70 + "\n")
    
    response = input("Continue with evaluation? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        success = run_evaluation()
        
        if success:
            print("\n" + "="*70)
            print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\n📁 Results saved to: {OUTPUT_DIR}")
            print("\n📊 Generated files:")
            print("  - metrics.json (all metrics)")
            print("  - roc_curve.png")
            print("  - precision_recall_curve.png")
            print("  - confusion_matrix.png")
            print("  - far_frr_curves.png")
            print("  - metrics_summary.png")
            print("  - score_distribution.png")
            print("  - per_class_metrics.png")
            print("\n🎓 All visualizations are 300 DPI, ready for your paper!")
            print("\n" + "="*70 + "\n")
    else:
        print("\n❌ Evaluation cancelled.")
        print("Please update the paths in this script and try again.")
