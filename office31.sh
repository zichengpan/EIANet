gpu=1
temperature=0.1
output="office31_weight"

python src_pretrain.py --dset a2d --gpu_id $gpu --office31 --output $output
python src_pretrain.py --dset a2w --gpu_id $gpu --office31 --output $output
python src_pretrain.py --dset d2a --gpu_id $gpu --office31 --output $output
python src_pretrain.py --dset d2w --gpu_id $gpu --office31 --output $output
python src_pretrain.py --dset w2a --gpu_id $gpu --office31 --output $output
python src_pretrain.py --dset w2d --gpu_id $gpu --office31 --output $output

python tar_adapt.py --dset w2a --gpu_id $gpu --office31 --output $output --temperature $temperature
python tar_adapt.py --dset a2w --gpu_id $gpu --office31 --output $output --temperature $temperature
python tar_adapt.py --dset d2w --gpu_id $gpu --office31 --output $output --temperature $temperature
python tar_adapt.py --dset a2d --gpu_id $gpu --office31 --output $output --temperature $temperature
python tar_adapt.py --dset d2a --gpu_id $gpu --office31 --output $output --temperature $temperature
python tar_adapt.py --dset w2d --gpu_id $gpu --office31 --output $output --temperature $temperature

