ABS_PATH=$(dirname $(realpath $0))
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold1/train" --test_dir="${ABS_PATH}/full_data/folds/fold1/test" --n_cuda_device=2 --seed=3 --n_epochs=80 --batch_size=32 
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold2/train" --test_dir="${ABS_PATH}/full_data/folds/fold2/test" --n_cuda_device=2 --seed=3 --n_epochs=80 --batch_size=32
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold3/train" --test_dir="${ABS_PATH}/full_data/folds/fold3/test" --n_cuda_device=2 --seed=3 --n_epochs=80 --batch_size=32
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold4/train" --test_dir="${ABS_PATH}/full_data/folds/fold4/test" --n_cuda_device=2 --seed=3 --n_epochs=80 --batch_size=32
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --train_dir="${ABS_PATH}/full_data/folds/fold5/train" --test_dir="${ABS_PATH}/full_data/folds/fold5/test" --n_cuda_device=2 --seed=3 --n_epochs=80 --batch_size=32
