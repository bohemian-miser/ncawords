#!/bin/bash
source venv/bin/activate

# Resume training for the ones that didn't finish

python -m nca.train_cloud --text COMP --steps 16000 --log-every 100 --snap-dir snaps_cloud

# Noise Seed variants
NOISE_START=1 python -m nca.train_web_method1 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method1_noise
NOISE_START=1 python -m nca.train_method2 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method2_noise
NOISE_START=1 python -m nca.train_web_method4 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method4_noise
NOISE_START=1 python -m nca.train_web_method5 --text COMP --steps 16000 --log-every 100 --snap-dir snaps_web_method5_noise
NOISE_START=1 python -m nca.train_web_9_line --text COMP --steps 16000 --log-every 100 --snap-dir snaps_9_line_noise

echo "All done!"
