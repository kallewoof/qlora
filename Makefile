dataset:
	make -C ../private-datasets/sbbpck-compiled
	python ./convert-raw-dataset.py
	rm -rf last_run_prepared

cabb_eval_dataset:
	make -C ../private-datasets/sbbpck-instruct/cabb
	cp ../private-datasets/sbbpck-instruct/cabb/curated.jsonl ./cabb-curated-dataset.jsonl
	rm -rf last_run_prepared_cabb_eval

axolotl_bin:
	cd ../axolotl && pip3 install -e '.[flash-attn,deepspeed]' --no-deps

13b: dataset axolotl_bin
	rm curr-model
	ln -s base-models/llama2-13b curr-model
	./writelog.sh 13b
	accelerate launch -m axolotl.cli.train ./axolotl-13b.yml | tee tlogs/latest/accelerate.log
	./posttrainlog.sh 13b

13b-from-6.1: dataset axolotl_bin
	rm -r qlora-out
	mkdir qlora-out
	cp -r input-loras/cf-v6.1-13b qlora-out/checkpoint-0
	make 13b

20b: dataset axolotl_bin
	rm curr-model
	ln -s base-models/llama2-emerhyst-20b curr-model
	./writelog.sh 20b
	accelerate launch -m axolotl.cli.train ./axolotl-20b.yml | tee tlogs/latest/accelerate.log
	./posttrainlog.sh 20b

70b-layered-256: dataset axolotl_bin
	python ./truncate-dataset.py 0.30
	rm curr-model
	ln -s base-models/llama2-70b curr-model
	./writelog.sh 70b
	python -m axolotl.cli.train ./axolotl-70b-256.yml | tee tlogs/latest/train-256.log
	./ttag.sh "70b-layered-01-256"
	./posttrainlog.sh 70b
	mkdir -p ./70b-layered-cps/256
	cp qlora-out/adapter* ./70b-layered-cps/256
	echo "{}" > ./70b-layered-cps/256/trainer-state.json
	make 70b-layered-512

70b-layered-512: dataset axolotl_bin
	python ./truncate-dataset.py 0.40
	rm curr-model
	ln -s base-models/llama2-70b curr-model
	./writelog.sh 70b
	python -m axolotl.cli.train ./axolotl-70b-512.yml | tee tlogs/latest/train-512.log
	./ttag.sh "70b-layered-02-512"
	./posttrainlog.sh 70b
	mkdir -p ./70b-layered-cps/512
	cp qlora-out/adapter* ./70b-layered-cps/512
	echo "{}" > ./70b-layered-cps/512/trainer-state.json
	make 70b-layered-1024

70b-layered-1024: dataset axolotl_bin
	python ./truncate-dataset.py 0.50
	rm curr-model
	ln -s base-models/llama2-70b curr-model
	./writelog.sh 70b
	python -m axolotl.cli.train ./axolotl-70b-1024.yml | tee tlogs/latest/train-1024.log
	./ttag.sh "70b-layered-03-1024"
	./posttrainlog.sh 70b
	mkdir -p ./70b-layered-cps/1024
	cp qlora-out/adapter* ./70b-layered-cps/1024
	echo "{}" > ./70b-layered-cps/1024/trainer-state.json
	make 70b-layered-final

70b-layered-final: dataset axolotl_bin
	# rm curr-model
	# ln -s base-models/llama2-70b curr-model
	./writelog.sh 70b
	python -m axolotl.cli.train ./axolotl-70b.yml | tee tlogs/latest/train-final.log
	./ttag.sh "70b-layered-04-final"
	./posttrainlog.sh 70b

70b-layered-final-xwin: dataset axolotl_bin
	# rm curr-model
	# ln -s base-models/llama2-70b curr-model
	./writelog.sh 70b
	python -m axolotl.cli.train ./axolotl-70b.yml | tee tlogs/latest/train-final.log
	./ttag.sh "70b-layered-04-final"
	./posttrainlog.sh 70b

70b: dataset axolotl_bin
	rm curr-model
	ln -s base-models/llama2-70b curr-model
	./writelog.sh 70b
	python -m axolotl.cli.train ./axolotl-70b.yml | tee tlogs/latest/train-final.log
	./posttrainlog.sh 70b

cabb-eval-70b: cabb_eval_dataset axolotl_bin
	rm curr-model
	ln -s base-models/llama2-70b curr-model
	./writelog.sh cabb-eval-70b
	python -m axolotl.cli.train ./axolotl-cabb-eval-70b.yml | tee tlogs/latest/train-cabb-eval.log
	./posttrainlog.sh cabb-eval-70b

derivative-models/%: FORCE
	rm curr-model
	ln -s $@ curr-model
	python -m axolotl.cli.merge_lora axolotl-$(shell echo $@ | rev | cut -c -3 | rev).yml --lora_model_dir=./curr-lora --load_in_8bit=False --load_in_4bit=False

derivative-cabb-models/%: FORCE
	rm curr-model
	ln -s $@ curr-model
	python -m axolotl.cli.merge_lora axolotl-cabb-eval-70b.yml --lora_model_dir=./cabb-eval-out --load_in_8bit=False --load_in_4bit=False


FORCE: ;

all: dataset axolotl_bin
