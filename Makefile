dataset:
	make -C ../private-datasets/sbbpck-compiled
	python ./convert-raw-dataset.py
	rm -rf last_run_prepared

axolotl_bin:
	cd ../axolotl && pip3 install -e '.[flash-attn,deepspeed]' --no-deps

13b: dataset axolotl_bin
	rm curr-model
	ln -s /usr/ssd/models/NousResearch_Llama2-13b-hf curr-model
	./writelog.sh 13b
	accelerate launch -m axolotl.cli.train ./axolotl-13b.yml | tee tlogs/latest/accelerate.log
	./posttrainlog.sh 13b

20b: dataset axolotl_bin
	rm curr-model
	ln -s /usr/ssd/models/Undi95_Emerhyst-20B curr-model
	./writelog.sh 20b
	accelerate launch -m axolotl.cli.train ./axolotl-20b.yml | tee tlogs/latest/accelerate.log
	./posttrainlog.sh 20b

70b: dataset axolotl_bin
	rm curr-model
	ln -s /usr/ssd/models/TheBloke_Llama-2-70B-fp16 curr-model
	./writelog.sh 70b
	# python -m axolotl.cli.train ./axolotl-70b-256.yml | tee tlogs/latest/train-256.log
	# python -m axolotl.cli.train ./axolotl-70b-512.yml | tee tlogs/latest/train-512.log
	python -m axolotl.cli.train ./axolotl-70b-1024.yml | tee tlogs/latest/train-1024.log
	python -m axolotl.cli.train ./axolotl-70b.yml | tee tlogs/latest/train-final.log
	./posttrainlog.sh 70b

all: dataset axolotl_bin
