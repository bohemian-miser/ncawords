#!/bin/bash
export OMP_NUM_THREADS=6

venv/bin/python -m nca.train_diffusion --text COMP --steps 16000 --log-every 100 --snap-dir snaps_diffusion_16k > diffusion.log 2>&1 &
venv/bin/python -m nca.train_cloud --text COMP --steps 16000 --log-every 100 --snap-dir snaps_cloud > cloud.log 2>&1 &

NOISE_START=1 venv/bin/python -m nca.train_web_method1 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method1_noise > method1.log 2>&1 &
NOISE_START=1 venv/bin/python -m nca.train_method2 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method2_noise > method2.log 2>&1 &
NOISE_START=1 venv/bin/python -m nca.train_web_method4 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method4_noise > method4.log 2>&1 &
NOISE_START=1 venv/bin/python -m nca.train_web_method5 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method5_noise > method5.log 2>&1 &
NOISE_START=1 venv/bin/python -m nca.train_web_9_line --text COMP --steps 16000 --log-every 100 --snap-dir snaps_9_line_noise > method9.log 2>&1 &

wait
