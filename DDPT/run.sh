#!/bin/bash

# Default values (can be overridden by command line arguments)
TRAINING="false"
EVALUATION="" 
DATASET="BraTS20"
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --Training)
      TRAINING=$(echo "$2" | tr '[:upper:]' '[:lower:]')
      shift 2
      ;;
    --Dataset)
      DATASET="$2"
      shift 2
      ;;
    --Evaluation)
      EVALUATION=$(echo "$2" | tr '[:upper:]' '[:lower:]')
      shift 2
      ;;
    --Checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--Training True/False] [--Dataset BraTS20/BraTS21/BraTS23/MSD] [--Evaluation True/False] [--Checkpoint /path/to/checkpoint]"
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ "$TRAINING" != "true" && "$TRAINING" != "false" ]]; then
  echo "Error: --Training must be True or False"
  exit 1
fi

if [[ "$EVALUATION" != "true" && "$EVALUATION" != "false" ]]; then
  echo "Error: --Evaluation must be True or False"
  exit 1
fi

# Validate dataset
case $DATASET in
  BraTS20|BraTS21|BraTS23|MSD)
    ;;
  *)
    echo "Error: --Dataset must be one of: BraTS20, BraTS21, BraTS23, MSD"
    exit 1
    ;;
esac

# Define required variables
MODEL="DPT"
EPOCHS=60
THRESHOLD=0.5
MEDIAN="False"
NUM_SHOTS=10000

# Check dataset structure
check_dataset_structure() {
  local datadir="../DATA/$1"
  local folder="$2"
  if [ ! -d "$datadir/$folder/Healthy" ] || [ ! -d "$datadir/$folder/Unhealthy" ]; then
    echo "Error: Invalid dataset structure for $1 in $folder"
    echo "Expected: $datadir/$folder/Healthy and $datadir/$folder/Unhealthy"
    exit 1
  fi
  echo "Dataset structure validated for $1 in $folder"
}

MODEL_DIR="models/${DATASET}/${NUM_SHOTS}_shots"
j=1
TRAIN_CRAFT="False"
EVAL_CRAFT="False"

if [[ "$TRAINING" == "true" ]]; then
  check_dataset_structure "$DATASET" "Training_images"
  echo "Starting training phase..."
  
  # Set training-specific parameters
  EVAL=False  # Don't run evaluation during training
  CLASSIFY=False
  
  # Resume from checkpoint if provided
  RESUME_ARG=""
  if [[ -n "$CHECKPOINT" ]]; then
    RESUME_ARG="--resume $CHECKPOINT"
    echo "Resuming training from checkpoint: $CHECKPOINT"
  fi
  
  if [[ -z "$RESUME_ARG" && -d "$MODEL_DIR" ]]; then
    echo "Clearing existing model directory $MODEL_DIR for fresh training"
    rm -rf "$MODEL_DIR"
  fi
  
  ROOT="../DATA/${DATASET}/Training_images"
  
  python DDPT-model.py --root $ROOT --seed $j --evaluate $EVAL --classify $CLASSIFY --train-craft $TRAIN_CRAFT --eval-craft $EVAL_CRAFT \
  $RESUME_ARG \
  --trainer $MODEL \
  --epochs $EPOCHS \
  --threshold $THRESHOLD \
  --median $MEDIAN \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --output-dir $MODEL_DIR \
  --config-file ./configs/trainers/VPT/vit_b32_deep.yaml \
  TRAINER.COOP.N_CTX 16 \
  TRAINER.COOP.CSC False \
  TRAINER.COOP.CLASS_TOKEN_POSITION end \
  DATASET.NUM_SHOTS $NUM_SHOTS \
  DATASET.TRAIN_PERCENT 0.6 \
  DATASET.VAL_PERCENT 0.2 \
  TRAINER.VPT.N_CTX 10 \
  TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT 12 \
  TRAINER.SELECTED_COVPT.CPN 10 \
  OPTIM.LR 0.01 \
  OPTIM.MAX_EPOCH 60 \
  PRETRAIN.C 30 \
  TRAINER.ALPHA 0.3
  
  if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
  fi
  echo "Training completed successfully!"
fi

# Evaluation phase
if [[ -n "$EVALUATION" ]] && [[ "$EVALUATION" == "true" ]]; then
  echo "Starting evaluation phase..."
  
  # Set evaluation-specific parameters
  EVAL=True
  CLASSIFY=False
  
  # Determine checkpoint path for evaluation
  EVAL_CHECKPOINT=""
  if [[ "$TRAINING" == "true" ]]; then
    # Use the model just trained
    EVAL_CHECKPOINT="$MODEL_DIR"
    echo "Using newly trained model for evaluation"
  elif [[ -n "$CHECKPOINT" ]]; then
    # Use provided checkpoint
    EVAL_CHECKPOINT="$CHECKPOINT"
    echo "Using provided checkpoint for evaluation: $CHECKPOINT"
  fi
  
  # Add resume argument if we have a checkpoint
  RESUME_ARG=""
  if [[ -n "$EVAL_CHECKPOINT" ]]; then
    RESUME_ARG="--resume $EVAL_CHECKPOINT"
  fi
  
  TRAIN_FOLDER="Training_images"
  ROOT_TRAIN="../DATA/${DATASET}/$TRAIN_FOLDER"
  if [ -d "$ROOT_TRAIN" ]; then
    check_dataset_structure "$DATASET" "$TRAIN_FOLDER"
    echo "Processing $TRAIN_FOLDER..."
    python DDPT-model.py --root $ROOT_TRAIN --seed $j --evaluate $EVAL --classify $CLASSIFY --train-craft $TRAIN_CRAFT --eval-craft $EVAL_CRAFT $RESUME_ARG --trainer $MODEL --epochs $EPOCHS --threshold $THRESHOLD --median $MEDIAN --dataset-config-file configs/datasets/${DATASET}.yaml --output-dir $MODEL_DIR --config-file ./configs/trainers/VPT/vit_b32_deep.yaml TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC False TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS $NUM_SHOTS DATASET.TRAIN_PERCENT 0.0 DATASET.VAL_PERCENT 0.2 TRAINER.VPT.N_CTX 10 TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT 12 TRAINER.SELECTED_COVPT.CPN 10 OPTIM.LR 0.01 OPTIM.MAX_EPOCH 60 PRETRAIN.C 30 TRAINER.ALPHA 0.3
    if [ $? -ne 0 ]; then
      echo "Evaluation on $TRAIN_FOLDER failed!"
      exit 1
    fi
    echo "Evaluation on $TRAIN_FOLDER completed successfully!"
  fi

  TEST_FOLDER="Testing_images"
  ROOT_TEST="../DATA/${DATASET}/$TEST_FOLDER"
  if [ -d "$ROOT_TEST" ]; then
    check_dataset_structure "$DATASET" "$TEST_FOLDER"
    echo "Processing $TEST_FOLDER..."
    python DDPT-model.py --root $ROOT_TEST --seed $j --evaluate $EVAL --classify $CLASSIFY --train-craft $TRAIN_CRAFT --eval-craft $EVAL_CRAFT $RESUME_ARG --trainer $MODEL --epochs $EPOCHS --threshold $THRESHOLD --median $MEDIAN --dataset-config-file configs/datasets/${DATASET}.yaml --output-dir $MODEL_DIR --config-file ./configs/trainers/VPT/vit_b32_deep.yaml TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC False TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS $NUM_SHOTS DATASET.TRAIN_PERCENT 0.0 DATASET.VAL_PERCENT 0.2 TRAINER.VPT.N_CTX 10 TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT 12 TRAINER.SELECTED_COVPT.CPN 10 OPTIM.LR 0.01 OPTIM.MAX_EPOCH 60 PRETRAIN.C 30 TRAINER.ALPHA 0.3
    if [ $? -ne 0 ]; then
      echo "Evaluation on $TEST_FOLDER failed!"
      exit 1
    fi
    echo "Evaluation on $TEST_FOLDER completed successfully!"
  fi
fi

# Summary
if [[ "$TRAINING" == "false" && "$EVALUATION" == "false" ]]; then
  echo "No training or evaluation requested. Use --Training True and/or --Evaluation True"
  exit 1
fi

echo "=== Process completed successfully ==="

