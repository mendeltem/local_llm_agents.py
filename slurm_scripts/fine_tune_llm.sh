#!/bin/bash
#SBATCH --job-name=fine_tune_llm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=sc-users
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uchralt.temuulen@charite.de
# 🛑 FIX: Using custom logs_dir for output/error
# %A is Job ID, %a is Array ID (useful if you switch to job arrays)
# Using full path to ensure logs go where expected
# The file name includes the script name for tracking
# Use /sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/fine_tune_log for POSIX path compatibility
# Use fine_tune_llm for output files
#SBATCH --output=/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/fine_tune_log/fine_tune_llm_%A_%a.out
#SBATCH --error=/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/fine_tune_log/fine_tune_llm_%A_%a.err


echo "Starting job fine_tune_llm on $(hostname) at $(date)"
echo "========================================="

# Initialize conda
__conda_setup="$('/opt/miniforge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniforge/etc/profile.d/conda.sh" ]; then
        . "/opt/miniforge/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniforge/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate environment
conda activate /sc-software/conda_envs/envs/gpulab-2025-2

# Load CUDA module 
echo "Loading module: devel/cuda/12.8"
# 🛑 FIX: Correct module load syntax
module load devel/cuda/12.8

# Check GPU status
nvidia-smi

echo "========================================="
echo "Executing Command: python /sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/fine_tune_llm.py"
# Execute the main command
python /sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/fine_tune_llm.py

echo "========================================="
echo "Job completed at $(date)"
