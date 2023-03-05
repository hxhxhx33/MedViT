include .env
include .env.local
export

BASE_ARGS=\
	--data_root "$(MEDVIT_DATA_ROOT)" \
	--fold_map "./resource/folds.json"


train:
	ARGS_FILES="arg/model.txt arg/runtime.txt arg/train.txt" \
	python -m cmd.train $(BASE_ARGS) \
		--folds 1 2 3 4 \
		--checkpoint_dir "$(MEDVIT_WORKSPACE)/checkpoint"

predict:
	ARGS_FILES="arg/model.txt arg/runtime.txt arg/predict.txt" \
	python -m cmd.predict $(BASE_ARGS) \
		--folds 0 \
		--checkpoint $(MEDVIT_WORKSPACE)/checkpoint/model-800.pt \
		--output_dir $(MEDVIT_WORKSPACE)/prediction

evaluate:
	python -m cmd.evaluate $(BASE_ARGS) \
		--folds 0 \
		--prediction_dir $(MEDVIT_WORKSPACE)/prediction
