# Cardiac T1 and T2 Mapping Segmentation
Source code for our paper "*Automated T1 and T2 mapping segmentation on cardiovascular magnetic resonance imaging using deep learning*".

### Setup Docker Container
For accessing the container by directly connecting to it via ssh: 
1. Create a keypair, copy the public key to the root of this repo and name it `cm-docker.pub`!
2. Run `make ssh`.
3. Connect on port 2233 `ssh root@<hostname> -i <private_key_paht> -p 2233`.

To run the container without starting an ssh server, run `make run`.

To customize docker build and run edit the Makefile.

> :warning: `make ssh` and `make run` starts the container with the `--rm` flag! Only contents of the `/workspace` persists if the container is stopped (via a simple volume mount)!

## How to run
### Supervised segmentation training

```
PYTHONPATH=. python supervised_segmentation/train.py -c supervised_segmentation/config.yaml
```

### Evaluation
Edit the `supervised_segmentation/inference.py` by adding your checkpoint path and run the following command:

```
PYTHONPATH=. python supervised_segmentation/inference.py
```