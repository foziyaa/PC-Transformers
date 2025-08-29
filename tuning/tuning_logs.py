def initialize_logs(study_name: str):
    """Create and initialize summary and trial log files."""
    trials_path = f"tuning/{study_name}_trials.txt"


    with open(trials_path, "w") as f:
        f.write(f"DETAILED TRIAL RESULTS - {study_name}\n")
        f.write(f"{'='*50}\n")
        f.write("Objective: Minimize Averge Energy \n\n")

#     return trials_path
# def initialize_logs(study_name: str):
#     """Create and initialize summary and trial log files."""
#     summary_path = f"tuning/{study_name}_summary.txt"
#     trials_path = f"tuning/{study_name}_trials.txt"

#     with open(summary_path, "w") as f:
#         f.write(f"BAYESIAN TUNING SUMMARY - {study_name}\n")
#         f.write(f"{'='*50}\n\n")
#         f.write("Objective: Minimize combined energy (normalized internal energy + CE loss)\n\n")
#         f.write("Trial Progress:\n")
#         f.write(f"{'Trial':<6} {'Time(s)':<8} {'CE Loss':<10} {'Raw Energy':<12} "
#                 f"{'Norm Energy':<12} {'Combined':<12} {'Energy Fn':<12}\n")
#         f.write(f"{'-'*82}\n")

#     with open(trials_path, "w") as f:
#         f.write(f"DETAILED TRIAL RESULTS - {study_name}\n")
#         f.write(f"{'='*50}\n")
#         f.write("Objective: Minimize combined energy (normalized internal energy + CE loss)\n\n")

#     return summary_path, trials_path

# def log_trial_to_summary(summary_path, trial):
#     """Appends a trial result to the summary log file."""
#     ce_loss = trial.user_attrs.get("ce_loss", "N/A")
#     energy = trial.user_attrs.get("energy", "N/A")
#     normalized_energy = trial.user_attrs.get("normalized_energy", "N/A")
#     combined_energy = trial.user_attrs.get("combined_energy", "N/A")
#     trial_time = trial.user_attrs.get("trial_time", 0)
#     config = trial.user_attrs.get("config", {})
#     energy_fn = config.get("energy_fn_name", "unknown")

#     with open(summary_path, "a") as f:
#         f.write(f"{trial.number:<6} {trial_time:<8.1f} {ce_loss:<10} {energy:<12} "
#                 f"{normalized_energy:<12} {combined_energy:<12} {energy_fn:<12}\n")
def log_trial_to_detailed_log(trials_path, trial, config, trial_time, avg_energy, write_header=False):
    """Appends trial information in tabular format to a trials log file."""
    with open(trials_path, "a") as f:
        if write_header:
            f.write(f"{'Trial':<6} | {'Time(s)':<8} | {'Avg Energy':<11} | "
                    f"{'n_embed':<7} | {'block_size':<10} | {'heads':<5} | {'blocks':<6} | {'T':<3} | "
                    f"{'LR':<8} | {'Warmup':<6} | {'Dropout':<7} | {'Bias':<5}\n")
            f.write("-" * 120 + "\n")
        
        f.write(f"{trial.number:<6} | {trial_time:<8.1f} | {avg_energy:<11.6f} | "
                f"{config.n_embed:<7} | {config.block_size:<10} | {config.num_heads:<5} | {config.n_blocks:<6} | "
                f"{config.T:<3} | {config.peak_learning_rate:<8.1e} | {config.warmup_steps:<6} | "
                f"{config.dropout:<7.3f} | {str(config.update_bias):<5}\n")
        
def write_final_results(results_path, trial):
    config = trial.user_attrs.get("config", {})
    energy = trial.user_attrs.get("energy", "N/A")

    with open(results_path, "w") as f:
        f.write("COMBINED ENERGY OPTIMIZATION RESULTS\n")
        f.write("====================================\n\n")
        f.write(f"Best combined energy: {trial.value:.4f}\n")
        f.write(f"Average Energy: {energy:.4f}\n")

        if config:
            f.write("Best Configuration:\n")
            for key, val in config.items():
                f.write(f"{key}: {val}\n")