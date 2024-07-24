gpu=0
epoch=40
scaler=0.5
output="cub_weight"

python src_pretrain.py --dset c2p --gpu_id $gpu --cub --output $output --batch_size 32
python src_pretrain.py --dset p2c --gpu_id $gpu --cub --output $output --batch_size 32

python tar_adapt.py --dset c2p --gpu_id $gpu --cub --output $output --batch_size 200 --lr_encoder $scaler --max_epoch $epoch
python tar_adapt.py --dset p2c --gpu_id $gpu --cub --output $output --batch_size 200 --lr_encoder $scaler --max_epoch $epoch

