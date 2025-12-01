python run_full.py --dataset datasets/can/ph/low_dim_v141.hdf5 --output_dir out/can/ --used_demo core_20 --task can --seeds 1 --cuda_device 0 --noise_scale 0.01

python run_full.py --dataset datasets/lift/ph/low_dim_v15.hdf5 --output_dir out/lift/ --used_demo core_20 --task lift --seeds 1 --cuda_device 0 --noise_scale 0.01

python run_full.py --dataset datasets/square/ph/low_dim_v141.hdf5 --output_dir out/square/ --used_demo core_20 --task square --seeds 1 --cuda_device 0 --noise_scale 0.01
