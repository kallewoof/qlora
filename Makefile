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
	accelerate launch -m axolotl.cli.train ./axolotl-13b.yml

20b: dataset axolotl_bin
	rm curr-model
	ln -s /usr/ssd/models/Undi95_Emerhyst-20B curr-model
	./writelog.sh 20b
	accelerate launch -m axolotl.cli.train ./axolotl-20b.yml

70b: dataset axolotl_bin
	rm curr-model
	ln -s /usr/ssd/models/TheBloke_Llama-2-70B-fp16 curr-model
	./writelog.sh 70b
	python -m axolotl.cli.train ./axolotl-70b.yml

all: dataset axolotl_bin
