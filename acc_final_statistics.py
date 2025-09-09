import os
import json
import pandas as pd

def is_fp(answer: str) -> bool:
    """
    Determine if a synthetic sample is a false positive (FP).
    """
    ans = answer.strip().lower()
    return ans == "no" or ans.startswith("no ")

def compute_accuracy(real_folder, synthetic_folder, output_csv_path):
    combined_results = []

    for filename in os.listdir(real_folder):
        if filename.endswith(".json") and filename in os.listdir(synthetic_folder):
            # Load real data
            with open(os.path.join(real_folder, filename), "r") as f:
                real_data = json.load(f)

            # Load synthetic data
            with open(os.path.join(synthetic_folder, filename), "r") as f:
                synth_data = json.load(f)

            # Real: Exo-centric (id ≤ 922) and Ego-centric (id > 922)
            real_exo = [x for x in real_data if int(x["id"].split("_")[-1]) <= 922]
            real_ego = [x for x in real_data if int(x["id"].split("_")[-1]) > 922]

            real_exo_correct = sum(x["correct"] for x in real_exo)
            real_ego_correct = sum(x["correct"] for x in real_ego)

            total_exo = len(real_exo)
            total_ego = len(real_ego)
            total_real = total_exo + total_ego

            real_exo_acc = real_exo_correct / total_exo * 100 if total_exo else 0
            real_ego_acc = real_ego_correct / total_ego * 100 if total_ego else 0
            real_overall_acc = (real_exo_correct + real_ego_correct) / total_real * 100 if total_real else 0

            # Synthetic: split by FP vs non-FP
            synth_fp = [x for x in synth_data if is_fp(x["ground_truth_answer"])]
            synth_nofp = [x for x in synth_data if not is_fp(x["ground_truth_answer"])]

            synth_fp_correct = sum(x["correct"] for x in synth_fp)
            synth_nofp_correct = sum(x["correct"] for x in synth_nofp)

            total_fp = len(synth_fp)
            total_nofp = len(synth_nofp)
            total_synth = total_fp + total_nofp

            synth_fp_acc = synth_fp_correct / total_fp * 100 if total_fp else 0
            synth_nofp_acc = synth_nofp_correct / total_nofp * 100 if total_nofp else 0
            synth_overall_acc = (synth_fp_correct + synth_nofp_correct) / total_synth * 100 if total_synth else 0

            # Overall weighted accuracy
            total_all = total_real + total_synth
            overall_acc = (
                (real_exo_correct + real_ego_correct + synth_fp_correct + synth_nofp_correct)
                / total_all * 100 if total_all else 0
            )

            combined_results.append({
                "Model (JSON File)": filename,
                "Real Ego-centric Accuracy (%)": round(real_ego_acc, 2),
                "Real Exo-centric Accuracy (%)": round(real_exo_acc, 2),
                "Real Overall Accuracy (%)": round(real_overall_acc, 2),
                "Synthetic Directional Accuracy (%)": round(synth_nofp_acc, 2),
                "Synthetic FP Accuracy (%)": round(synth_fp_acc, 2),
                "Synthetic Overall Accuracy (%)": round(synth_overall_acc, 2),
                "Overall Accuracy (%)": round(overall_acc, 2)
            })

    # Save to CSV
    df = pd.DataFrame(combined_results)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved results to: {output_csv_path}")

# === Example usage ===
os.makedirs("csv_final_results", exist_ok=True)

# To test with your own folders, set your paths below:
real_data_folder = "processed_outputs/real_mc_cot" 
synthetic_data_folder = "processed_outputs/synthetic_mc_cot"
output_csv = "csv_final_results/final_accuracy_table_cot.csv"


############### To reproduce numbers in the paper, use the following folders ###############
# You can uncomment one of the two sections below.

# Uncomment this part for CoT results (Table 1 in the paper)
# real_data_folder = "processed_outputs_paper_results/real_mc_cot"
# synthetic_data_folder = "processed_outputs_paper_results/synthetic_mc_cot"
# output_csv = "csv_final_results/final_accuracy_table_cot.csv"

# Or uncomment this part for Direct output results (Figure 6 in the paper)
# real_data_folder = "processed_outputs_paper_results/real_mc_direct-output"
# synthetic_data_folder = "processed_outputs_paper_results/synthetic_mc_direct-output"
# output_csv = "csv_final_results/final_accuracy_table_do.csv"

############################################################################################

compute_accuracy(real_data_folder, synthetic_data_folder, output_csv)