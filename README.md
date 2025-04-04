Code for **MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences** (https://arxiv.org/pdf/2402.08925)

Tulu-7b, Tulu-2-7b and llama2-7b can be used in this codebase. Change the datapath and model path in the scripts.

Data(from RLPHF): https://drive.google.com/drive/folders/1uyW6i1bmUF9UO5kCFvcW3AJhmKhOewr4?usp=sharing

# Reward Learning with EM

See examples in EM_TULU.slurm

# Max Min RLHF Alignment
## Reward Training
Run job_train_seperate.slurm and job_biased_single_reward_model.slurm
## PPO
Run jobrlhf.slurm for baseline and run jobrrlhf.slurm for max min RLHF 
## Generation
Run job_generate1.slurm or job_generate1_llama2_P1_maxmin.slurm as examples
## GPT4 Evaluation
Run GPT4_evaluation.py

Reference: https://github.com/joeljang/RLPHF
