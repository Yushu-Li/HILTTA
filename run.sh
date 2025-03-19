#!/bin/bash

# Define method and dataset arrays
methods=("pl" "tent" "shot" "eata" "sar" "rmt")
datasets=("cifar10-c" "cifar100-c" "imagenet-c" "imagenet-d")

# Define other variables
selections=("THRESHOLD" "LR" "THETA" "D_MARGIN" "LR" "LAMBDA_CONT")
selection_choices=(
  "0.05,0.1,0.2,0.4,0.6,0.8,1.0"
  "5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,5e-3"
  "0.1,0.25,0.5,1.0,2.5,5.0,10.0"
  "0.05,0.1,0.2,0.4,0.6,0.8,1.0"
  "5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,5e-3"
  "0.1,0.25,0.5,1.0,2.5,5.0,10.0"
)

# Use getopts to process command-line arguments
while getopts "m:d:" opt; do
  case $opt in
    m) method="$OPTARG" ;;
    d) dataset="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Check if method and dataset are provided
if [[ -z "$method" || -z "$dataset" ]]; then
  echo "Usage: $0 -m <method> -d <dataset>"
  echo "Methods: ${methods[*]}"
  echo "Datasets: ${datasets[*]}"
  exit 1
fi

# Check if method and dataset are valid
if [[ ! " ${methods[*]} " =~ " ${method} " ]]; then
  echo "Invalid method: $method"
  echo "Valid methods: ${methods[*]}"
  exit 1
fi

if [[ ! " ${datasets[*]} " =~ " ${dataset} " ]]; then
  echo "Invalid dataset: $dataset"
  echo "Valid datasets: ${datasets[*]}"
  exit 1
fi

# Determine the index of method
method_index=-1
for i in "${!methods[@]}"; do
  if [[ "${methods[$i]}" == "$method" ]]; then
    method_index=$i
    break
  fi
done

# Determine the index of dataset
dataset_index=-1
for i in "${!datasets[@]}"; do
  if [[ "${datasets[$i]}" == "$dataset" ]]; then
    dataset_index=$i
    break
  fi
done


# Determine selection and selection_choice
selection="${selections[$method_index]}"
selection_choice="${selection_choices[$method_index]}"

# Construct config file path
cfg_path=""
case $dataset in
  cifar10-c) cfg_path="cfgs/cifar10_c/${method}.yaml" ;;
  cifar100-c) cfg_path="cfgs/cifar100_c/${method}.yaml" ;;
  imagenet-c) cfg_path="cfgs/imagenet_c/${method}.yaml" ;;
  imagenet-d) cfg_path="cfgs/imagenet_others/${method}.yaml" ;;
esac

# Construct save_dir
save_dir="./output/"

# Construct python command
python_command="python main.py --cfg $cfg_path SAVE_DIR $save_dir ACTIVE.MODEL_SELECTION $selection ACTIVE.MODEL_SELECTION_CHOICE '$selection_choice'"

# Add extra parameters based on dataset
case $dataset in
  imagenet-c) python_command="$python_command CORRUPTION.NUM_EX 5000" ;;
  imagenet-d) python_command="$python_command CORRUPTION.NUM_EX 10000 CORRUPTION.DATASET imagenet_d109" ;;
esac

# Execute python command
echo "Running: $python_command"
eval "$python_command"