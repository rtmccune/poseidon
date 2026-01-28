import argparse
import logging
from mpi4py import MPI
import poseidon_core

def main():
    # REQUIRED for HPC
    MPI.Init()
    
    parser = argparse.ArgumentParser(description="Analyze roadway flood depths along a LabelMe line.")
    
    parser.add_argument("--event_dir", type=str, required=True, help="Path to flood events directory")
    parser.add_argument("--json_path", type=str, required=True, help="Path to LabelMe JSON file with roadway line")
    parser.add_argument("--label", type=str, default="roadway", help="Label name of the line in JSON (default: roadway)")
    parser.add_argument("--step_size", type=float, default=1.0, help="Interpolation step size in pixels (default: 1.0)")

    parser.add_argument("--statistic", type=str, default="95_perc", 
                        choices=["95_perc", "90_perc", "mean", "median"],
                        help="Statistic to analyze (default: 95_perc)")
    
    args = parser.parse_args()

    # Initialize
    analyzer = poseidon_core.RoadwayAnalyzer(
        main_dir=args.event_dir,
        labelme_json_path=args.json_path,
        line_label=args.label,
        step_size=args.step_size
        statistic=args.statistic
    )

    # Run Pipeline
    analyzer.run_hpc_pipeline()

if __name__ == "__main__":
    main()