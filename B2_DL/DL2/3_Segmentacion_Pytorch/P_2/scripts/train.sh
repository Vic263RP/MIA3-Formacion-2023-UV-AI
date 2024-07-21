# Antonio Martínez González

LOGS_DIR="logs" # Directory where training logs will be saved. In this setting, the directory will contain models and log files.

# This if-else block creates the logs directory if it does not exist
if [ ! -d "$LOGS_DIR" ]; then
    echo "Directory $LOGS_DIR does not exist. Creating directory"
    mkdir -p "$LOGS_DIR"
    echo "Directory created"
fi

# Train model
python3 train.py --image_train_directory "../dataset/train/images" \
                 --mask_train_directory "../dataset/train/masks" \
                 --image_test_directory "../dataset/val/images" \
                 --mask_test_directory "../dataset/val/masks" \
                 --mixed_precision_training True \
                 --in_channels 3 \
                 --out_channels 1 \
                 --batch_norm True \
                 --n_epochs 10 \
                 --learning_rate "0.0001" \
                 --batch_size 16 \
                 --n_workers 1 \
                 --shuffle True \
                 --crop True \
                 --image_height 160 \
                 --image_width 240 \
                 --load_pretrained "" \
                 --model_path "models/testingscriptsv0.pth" \
                 --early_stopping True \
                 --patience 3 \
                 --min_delta 0.0 \
                 --mode "max" \
                 --predictions_dir "predictions/testingscriptsv0";

# The two following lines show how this sh file is intended to be used
# cat train.sh > path-to-log-file.txt # write the contents of the sh file on a txt log file 'path-to-log-file.txt'
# . train.sh >> path-to-log-file.txt 2>&1 # appends stdout+errors resulting from the execution of the sh file to the txt log file 'path-to-log-file.txt'