#!/bin/bash
source venv/bin/activate
export OMP_NUM_THREADS=4
mkdir -p logs

cat << 'EOF' > dyn_cmds.txt
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 1 --support-vol 0.7 --snap-dir snaps_dyn_clear_n1_vol0_7 > logs/snaps_dyn_clear_n1_vol0_7.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 1 --support-vol 0.4 --snap-dir snaps_dyn_clear_n1_vol0_4 > logs/snaps_dyn_clear_n1_vol0_4.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 1 --support-vol 0.1 --snap-dir snaps_dyn_clear_n1_vol0_1 > logs/snaps_dyn_clear_n1_vol0_1.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 100 --support-vol 0.7 --snap-dir snaps_dyn_clear_n100_vol0_7 > logs/snaps_dyn_clear_n100_vol0_7.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 100 --support-vol 0.4 --snap-dir snaps_dyn_clear_n100_vol0_4 > logs/snaps_dyn_clear_n100_vol0_4.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 100 --support-vol 0.1 --snap-dir snaps_dyn_clear_n100_vol0_1 > logs/snaps_dyn_clear_n100_vol0_1.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 500 --support-vol 0.7 --snap-dir snaps_dyn_clear_n500_vol0_7 > logs/snaps_dyn_clear_n500_vol0_7.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 500 --support-vol 0.4 --snap-dir snaps_dyn_clear_n500_vol0_4 > logs/snaps_dyn_clear_n500_vol0_4.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 500 --support-vol 0.1 --snap-dir snaps_dyn_clear_n500_vol0_1 > logs/snaps_dyn_clear_n500_vol0_1.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 1000 --support-vol 0.7 --snap-dir snaps_dyn_clear_n1000_vol0_7 > logs/snaps_dyn_clear_n1000_vol0_7.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 1000 --support-vol 0.4 --snap-dir snaps_dyn_clear_n1000_vol0_4 > logs/snaps_dyn_clear_n1000_vol0_4.log 2>&1
venv/bin/python -m nca.train_dynamic_organic --text COMP --steps 16000 --log-every 100 --no-noise --update-every 1000 --support-vol 0.1 --snap-dir snaps_dyn_clear_n1000_vol0_1 > logs/snaps_dyn_clear_n1000_vol0_1.log 2>&1
EOF

echo "Starting 3 parallel workers for 12 jobs..."
xargs -d '\n' -P 3 -I CMD bash -c 'CMD' < dyn_cmds.txt
echo "All 12 jobs have completed (or failed)."
