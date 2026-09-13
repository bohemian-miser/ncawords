Local GPU Acceleration Export

Instructions for your laptop:
1. Extract this tarball: `tar -xvf nca_laptop_export.tar.gz`
2. Create python environment: `python3 -m venv venv && source venv/bin/activate`
3. Install reqs (if you haven't): `pip install torch torchvision numpy Pillow`
4. Run the automated sequential trainer: `python laptop_train.py`

This will detect your CUDA/MPS hardware seamlessly and churn through the runs. 
Once they reach step 5000, you can just merge/scp the `nca_runs/` folder back up to the server here and sync to bucket!
