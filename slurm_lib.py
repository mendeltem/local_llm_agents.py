import os
import subprocess
import textwrap
from pathlib import Path
from typing import Union, Optional

def submit_slurm_job(
    cmd_str: str,
    job_name: str,
    log_dir: Union[str, Path],
    slurm_dir: Union[str, Path],
    time_limit: str = "02:00:00",
    memory: str = "32G",
    cuda_module: str = "devel/cuda/12.8",
    conda_env_path: str = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/Software/miniconda3/envs/microbleednet",
    cpus: int = 8,
    gpus: str = "shard:1",
    account: str = "sc-users",
    mail_user: Optional[str] = "uchralt.temuulen@charite.de"
) -> Optional[str]:
    """
    Creates a SLURM script and submits a job to the cluster.

    Args:
        cmd_str: The full command string to execute (e.g., 'microbleednet fine_tune ...').
        job_name: Base name for the SLURM job and script file.
        log_dir: Directory where the SLURM output/error logs will be written.
        slurm_dir: Directory where the temporary SLURM script file will be saved.
        time_limit: Time limit for the job (e.g., "02:00:00").
        memory: Memory requested (e.g., "32G").
        cuda_module: The environment module for CUDA to load.
        conda_env_path: The full path to the Conda environment to activate.
        cpus: Number of CPUs/cores requested.
        account: The SLURM account name.
        mail_user: Email address for job alerts (None to disable).

    Returns:
        The SLURM Job ID if submission is successful, otherwise None.
    """
    
    # 1. Prepare directories and filenames
    log_dir = Path(log_dir).resolve()
    slurm_dir = Path(slurm_dir).resolve()
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(slurm_dir, exist_ok=True)
    
    script_file = slurm_dir / f"{job_name}.sh"
    
    mail_directives = ""
    if mail_user:
        mail_directives = f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={mail_user}"
        
    # 2. Define the SLURM script content
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres={gpus}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --account={account}
{mail_directives}
# 🛑 FIX: Using custom logs_dir for output/error
# %A is Job ID, %a is Array ID (useful if you switch to job arrays)
# Using full path to ensure logs go where expected
# The file name includes the script name for tracking
# Use {log_dir.as_posix()} for POSIX path compatibility
# Use {job_name} for output files
#SBATCH --output={log_dir.as_posix()}/{job_name}_%A_%a.out
#SBATCH --error={log_dir.as_posix()}/{job_name}_%A_%a.err


echo "Starting job {job_name} on $(hostname) at $(date)"
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
conda activate {conda_env_path}

# Load CUDA module 
echo "Loading module: {cuda_module}"
# 🛑 FIX: Correct module load syntax
module load {cuda_module}

# Check GPU status
nvidia-smi

echo "========================================="
echo "Executing Command: {cmd_str}"
# Execute the main command
{cmd_str}

echo "========================================="
echo "Job completed at $(date)"
"""
    # Use dedent to remove leading whitespace from the Python file definition
    slurm_script = textwrap.dedent(slurm_script_content)

    # 3. Write and set permissions
    try:
        with open(script_file, "w") as f:
            f.write(slurm_script)
        os.chmod(script_file, 0o755)
    except Exception as e:
        print(f"❌ Error writing SLURM script {script_file}: {e}")
        return None

    # 4. Submit job
    print(f"\n👉 Submitting job {job_name}...")
    try:
        # Use full path to sbatch or rely on PATH if reliable
        result = subprocess.run(["sbatch", script_file.as_posix()], capture_output=True, text=True, check=True)
        
        job_id = result.stdout.strip().split()[-1] # Usually returns "Submitted batch job XXXXX"
        print(f"✅ Success! Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"❌ Error submitting SLURM job {job_name}:")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"❌ Unexpected error during submission: {e}")
        return None