ABS_PATH=$(dirname $(realpath $0))
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold1/train" --test_dir="${ABS_PATH}/full_data/folds/fold1/test"
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold2/train" --test_dir="${ABS_PATH}/full_data/folds/fold2/test"
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold3/train" --test_dir="${ABS_PATH}/full_data/folds/fold3/test"
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold4/train" --test_dir="${ABS_PATH}/full_data/folds/fold4/test"
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATh}/full_data/folds/fold5/train" --test_dir="${ABS_PATH}/full_data/folds/fold5/test"
