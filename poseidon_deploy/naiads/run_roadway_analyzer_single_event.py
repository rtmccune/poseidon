import argparse
import sys
import os
import logging
from datetime import datetime
import poseidon_core

def _log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Analyze roadway flood depths along a LabelMe line.")
    
    parser.add_argument("--target_event_dir", type=str, required=True, help="Path to specific flood event directory")
    parser.add_argument("--json_path", type=str, required=True, help="Path to LabelMe JSON file with roadway line")
    parser.add_argument("--label", type=str, default="roadway", help="Label name of the line in JSON")
    parser.add_argument("--step_size", type=float, default=1.0, help="Interpolation step size in pixels")
    parser.add_argument("--statistic", type=str, default="95_perc", 
                        choices=["95_perc", "90_perc", "mean", "median"])
    
    args = parser.parse_args()

    # Initialize
    analyzer = poseidon_core.RoadwayAnalyzer(
        target_event_dir=args.target_event_dir,
        labelme_json_path=args.json_path,
        line_label=args.label,
        step_size=args.step_size,
        statistic=args.statistic
    )

    # Run Pipeline for the single event
    analyzer.process_single_event()

if __name__ == "__main__":
    main()
