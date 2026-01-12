import re
from pathlib import Path
import yaml

input_dir = Path("src/ensemble_trainer/old_codes")
py_files = sorted(input_dir.glob("CIFAR*.py"))
model_data = []

for file in py_files:
    content = file.read_text()
    model_splits = re.split(r'#\s*MODEL(\d+)', content)

    for i in range(1, len(model_splits), 2):
        model_id = f"MODEL{model_splits[i]}"
        model_code = model_splits[i + 1]

        # Extract only model definition section
        model_body_match = re.search(r'model\s*=\s*Sequential\(\)\s*(.*?)model\.compile\(', model_code, re.DOTALL)
        model_body = model_body_match.group(1) if model_body_match else ""

        # Conv filters
        conv_filters = [int(n) for n in re.findall(r'Conv2D\((\d+),', model_body)]

        # Pooling types
        pooling_types = []
        pooling_matches = re.findall(r'(AveragePooling2D|MaxPooling2D)', model_body)
        for pooling in pooling_matches:
            pooling_types.append('avg' if 'Average' in pooling else 'max')

        # Dense units
        dense_units = [int(n) for n in re.findall(r'Dense\((\d+)', model_body)]

        # Dropout rates
        dropout_rates = re.findall(r'Dropout\(([\d\.]+)', model_body)

        # Optimizer & learning rate
        optimizer = 'adam' if 'Adam' in model_code else 'sgd' if 'SGD' in model_code else 'unknown'
        lr_match = re.search(r'learning_rate\s*=\s*([\d\.]+)', model_code)
        learning_rate = float(lr_match.group(1)) if lr_match else None

        # Epochs
        epoch_match = re.search(r'fit\([^)]*epochs\s*=\s*(\d+)', model_code)
        epochs = int(epoch_match.group(1)) if epoch_match else None

        model_data.append({
            "model_id": model_id,
            "model_number": int(model_id.replace("MODEL", "")),
            "file": file.name,
            "conv_layer_count": len(conv_filters),
            "conv_filters": conv_filters,
            "pooling_types": pooling_types,
            "dense_layer_count": len(dense_units),
            "dense_units": dense_units,
            "dropout_rates": dropout_rates,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "epochs": epochs
        })

# Sort and write to YAML
sorted_model_data = sorted(model_data, key=lambda x: x["model_number"])
yaml_dict = {entry["model_id"]: {k: v for k, v in entry.items() if k != "model_id"} for entry in sorted_model_data}

yaml_output_path = Path("config/ensemble_config.yaml")
with open(yaml_output_path, "w") as f:
    yaml.dump(yaml_dict, f, sort_keys=False)


# df = pd.DataFrame(model_data)
# df["model_number"] = pd.to_numeric(df["model_number"], errors='coerce')
# df = df.sort_values(by="model_number", key=lambda x: x.astype(int), ascending=True)
# df.to_csv("src/ensemble_trainer/ensemble_hyperparameters.csv", index=False)
