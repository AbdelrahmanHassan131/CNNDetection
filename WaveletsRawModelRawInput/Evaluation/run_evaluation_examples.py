"""
Quick reference script for running WaveletsRawModelRawInput model evaluation
Copy and modify the command below to evaluate your model
"""

# Example 1: Basic evaluation with checkpoint file
# python evaluate_model.py \
#     --dataroot "G:/datasets/validation" \
#     --checkpoint "../checkpoints/wavelet_raw_experiment/latest_net.pth" \
#     --output_dir "./results/wavelet_raw_eval"

# Example 2: Full evaluation with all options
# python evaluate_model.py \
#     --dataroot "G:/datasets/validation" \
#     --checkpoint "../checkpoints/wavelet_raw_experiment/latest_net.pth" \
#     --name "wavelet_raw_experiment" \
#     --checkpoints_dir "../checkpoints" \
#     --output_dir "./results/wavelet_raw_eval" \
#     --batch_size 32 \
#     --threshold 0.5 \
#     --gpu_id 0

# Example 3: Using epoch name instead of full path
# python evaluate_model.py \
#     --dataroot "G:/datasets/validation" \
#     --checkpoint "latest" \
#     --name "wavelet_raw_experiment" \
#     --checkpoints_dir "../checkpoints" \
#     --output_dir "./results/wavelet_raw_eval"

# Example 4: CPU evaluation (no GPU)
# python evaluate_model.py \
#     --dataroot "G:/datasets/validation" \
#     --checkpoint "../checkpoints/wavelet_raw_experiment/latest_net.pth" \
#     --output_dir "./results/wavelet_raw_eval" \
#     --gpu_id -1

# Example 5: Custom threshold evaluation
# python evaluate_model.py \
#     --dataroot "G:/datasets/validation" \
#     --checkpoint "../checkpoints/wavelet_raw_experiment/latest_net.pth" \
#     --output_dir "./results/wavelet_raw_eval" \
#     --threshold 0.45

print("This is a reference script. Uncomment and modify the commands above to run evaluation.")
print("\nQuick start:")
print("1. Update --dataroot with your validation dataset path")
print("2. Update --checkpoint with your model checkpoint path")
print("3. Update --output_dir with desired output directory")
print("4. Run the command in terminal")
