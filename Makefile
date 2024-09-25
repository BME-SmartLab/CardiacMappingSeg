host_ssh_port=2233
image_name=cardacmapping
container_name=cardacmapping
data_path=/mnt/hdd2/se

run: build
	nvidia-docker run \
	-it --rm \
	--shm-size 1G \
	--network host \
	-e NVIDIA_VISIBLE_DEVICES=1 \
	--name $(container_name) \
	-v $(shell pwd):/workspace \
	-v $(data_path):/data/se:ro \
	$(image_name) \
	/bin/bash

build:
	docker build --tag $(image_name)  .

push:
	docker push $(image_name) 

pull:
	docker pull $(image_name) 

stop:
	docker stop $(container_name)

tensorboard: build
	docker run \
	-it --rm \
	--network host \
	-p $(host_tensorboard_port):6006 \
	--name tensorboard \
	-v $(shell pwd):/workspace \
	$(image_name) \
	tensorboard --logdir artifacts/ --reload_multifile True --bind_all

ssh: build
	nvidia-docker run \
	-dt --rm \
	--shm-size 1G \
	-p $(host_ssh_port):22 \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	--name $(container_name) \
	-v $(shell pwd):/workspace \
	-v $(data_path):/data/se:ro \
	$(image_name) \
	/usr/sbin/sshd -D