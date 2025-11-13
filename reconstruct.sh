python sample_reconstruction.py --use-cache --load-model-folder model_benchmark
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch200
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch400
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch600
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch800
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch1000
python sample_reconstruction.py --use-cache --load-model-folder model_downscale_32 -d --downscale-resolution 32
python sample_reconstruction.py --use-cache --load-model-folder model_downscale_64 -d --downscale-resolution 64
python sample_reconstruction.py --use-cache --load-model-folder model_downscale_96 -d --downscale-resolution 96
python sample_reconstruction.py --use-cache --load-model-folder model_pca --use-pca
python sample_reconstruction.py --use-cache --load-model-folder model_savgol_pca --use-savgol --use-pca
python sample_reconstruction.py --use-cache --load-model-folder model_wavelet_pca --use-wavelet --use-pca
python sample_reconstruction.py --use-cache --load-model-folder model_circles1
python sample_reconstruction.py --use-cache --load-model-folder model_circles2
python sample_reconstruction.py --use-cache --load-model-folder model_circles3
python sample_reconstruction.py --use-cache --load-model-folder model_circles4
python sample_reconstruction.py --use-cache --load-model-folder model_one_forth --use-subset --subset-percentage 0.25
python sample_reconstruction.py --use-cache --load-model-folder model_one_half --use-subset --subset-percentage 0.5
python sample_reconstruction.py --use-cache --load-model-folder model_three_forth --use-subset --subset-percentage 0.75
python sample_reconstruction.py --use-cache --load-model-folder model_skip_every_2 --skip-pins --skip-every 2
python sample_reconstruction.py --use-cache --load-model-folder model_skip_every_4 --skip-pins --skip-every 4