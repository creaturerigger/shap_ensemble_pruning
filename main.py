import yaml
import argparse
from src.flow.training_flow import training_flow

def load_config(config_path: str) -> dict:
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)

def main():
    parser = argparse.ArgumentParser(description="Run prefect training flow with custom config override.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file.")
    parser.add_argument("--datasets", nargs="+", required=True, help="List of dataset names to run.")

    args = parser.parse_args()
    config = load_config(args.config)

    for dataset_name in args.datasets:
        if dataset_name not in config:
            print(f"❌ Config for dataset '{dataset_name}' not found in the config file.")
            continue

        experiment = config[dataset_name]
        epochs = experiment.get("num_epochs", 5)

        print(f"\n🚀 Running flow with: dataset={dataset_name}, epochs={epochs}")
        training_flow(dataset_name=dataset_name, num_epochs=epochs)

if __name__ == "__main__":
    main()
