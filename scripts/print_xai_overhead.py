#!/usr/bin/env python3
import json
import glob
import os

def main():
    rate_input = 0.75 / 1_000_000
    rate_output = 4.50 / 1_000_000

    # Locate workspace paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_path_base = os.path.abspath(os.environ.get("WS_VANILLA", os.path.join(script_dir, "..", "ws_baseline")))
    workspace_path_xai = os.path.abspath(os.environ.get("WS_OURS", os.path.join(script_dir, "..", "ws_baseline_xai")))
    
    xai_pattern = os.path.join(workspace_path_xai, "*", "xai_overhead.json")
    base_pattern = os.path.join(workspace_path_base, "*", "xai_overhead.json")
    
    xai_files = glob.glob(xai_pattern)
    base_files = glob.glob(base_pattern)
    
    task_names = set()
    for f in xai_files:
        task_names.add(os.path.basename(os.path.dirname(f)))
    for f in base_files:
        task_names.add(os.path.basename(os.path.dirname(f)))
        
    sorted_tasks = sorted(list(task_names))

    if not sorted_tasks:
        print(f"No xai_overhead.json files found in {workspace_path_xai} or {workspace_path_base}")
        return

    # Print Header
    header_fmt = "{:<58} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:>12}"
    divider = "-" * 137
    
    print("Workspace Cost and XAI Overhead Report")
    print(f"Rates: Input (Prompt) = $0.75/M tokens, Output (Completion) = $4.50/M tokens")
    print(divider)
    print(header_fmt.format(
        "Task (Proxy Dataset)", 
        "XAI Prompt", 
        "XAI Compl.", 
        "Tot Prompt", 
        "Tot Compl.", 
        "XAI Cost ($)", 
        "Tot Cost ($)"
    ))
    print(divider)

    base_prompt_sum = 0
    base_completion_sum = 0
    xai_prompt_sum = 0
    xai_completion_sum = 0
    total_prompt_sum = 0
    total_completion_sum = 0
    num_base_tasks = 0
    num_xai_tasks = 0

    for task_name in sorted_tasks:
        # 1. Baseline Row (if exists)
        base_file = os.path.join(workspace_path_base, task_name, "xai_overhead.json")
        has_base = os.path.exists(base_file)
        if has_base:
            try:
                with open(base_file, 'r') as fh:
                    bdata = json.load(fh)
                bp = bdata.get("total_prompt_tokens", 0)
                bc = bdata.get("total_completion_tokens", 0)
                base_cost = (bp * rate_input) + (bc * rate_output)
                
                print(header_fmt.format(
                    f"{task_name} (Baseline)",
                    "0",
                    "0",
                    f"{bp:,}",
                    f"{bc:,}",
                    "0.000000",
                    f"{base_cost:.6f}"
                ))
                base_prompt_sum += bp
                base_completion_sum += bc
                num_base_tasks += 1
            except Exception as e:
                print(f"Error reading baseline {base_file}: {e}")

        # 2. XAI Row (if exists)
        xai_file = os.path.join(workspace_path_xai, task_name, "xai_overhead.json")
        has_xai = os.path.exists(xai_file)
        if has_xai:
            try:
                with open(xai_file, 'r') as fh:
                    xdata = json.load(fh)
                xp = xdata.get("xai_prompt_tokens", 0)
                xc = xdata.get("xai_completion_tokens", 0)
                tp = xdata.get("total_prompt_tokens", 0)
                tc = xdata.get("total_completion_tokens", 0)
                xai_cost = (xp * rate_input) + (xc * rate_output)
                total_cost = (tp * rate_input) + (tc * rate_output)
                
                print(header_fmt.format(
                    f"{task_name} (XAI)",
                    f"{xp:,}",
                    f"{xc:,}",
                    f"{tp:,}",
                    f"{tc:,}",
                    f"{xai_cost:.6f}",
                    f"{total_cost:.6f}"
                ))
                xai_prompt_sum += xp
                xai_completion_sum += xc
                total_prompt_sum += tp
                total_completion_sum += tc
                num_xai_tasks += 1
            except Exception as e:
                print(f"Error reading XAI {xai_file}: {e}")

    print(divider)
    base_total_cost = (base_prompt_sum * rate_input) + (base_completion_sum * rate_output)
    xai_total_cost = (xai_prompt_sum * rate_input) + (xai_completion_sum * rate_output)
    total_consume_cost = (total_prompt_sum * rate_input) + (total_completion_sum * rate_output)
    
    print(header_fmt.format(
        "TOTAL (Baseline)",
        "0",
        "0",
        f"{base_prompt_sum:,}",
        f"{base_completion_sum:,}",
        "0.000000",
        f"{base_total_cost:.6f}"
    ))
    print(header_fmt.format(
        "TOTAL (XAI)",
        f"{xai_prompt_sum:,}",
        f"{xai_completion_sum:,}",
        f"{total_prompt_sum:,}",
        f"{total_completion_sum:,}",
        f"{xai_total_cost:.6f}",
        f"{total_consume_cost:.6f}"
    ))
    print(divider)
    
    grand_prompt_sum = base_prompt_sum + total_prompt_sum
    grand_completion_sum = base_completion_sum + total_completion_sum
    grand_total_cost = base_total_cost + total_consume_cost
    
    print(header_fmt.format(
        "GRAND TOTAL",
        f"{xai_prompt_sum:,}",
        f"{xai_completion_sum:,}",
        f"{grand_prompt_sum:,}",
        f"{grand_completion_sum:,}",
        f"{xai_total_cost:.6f}",
        f"{grand_total_cost:.6f}"
    ))
    print(divider)

    base_avg_cost = (base_total_cost / num_base_tasks) if num_base_tasks > 0 else 0.0
    xai_avg_cost = (total_consume_cost / num_xai_tasks) if num_xai_tasks > 0 else 0.0

    print(f"Cost per task: Without XAI = ${base_avg_cost:.6f} ({num_base_tasks} tasks) | With XAI = ${xai_avg_cost:.6f} ({num_xai_tasks} tasks)")

if __name__ == "__main__":
    main()
