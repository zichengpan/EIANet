gpu=0
batchsize=128
scaler=0.5
output="office_home_weight"

python src_pretrain.py --dset p2c --gpu_id $gpu --home --output $output
python src_pretrain.py --dset p2r --gpu_id $gpu --home --output $output
python src_pretrain.py --dset p2a --gpu_id $gpu --home --output $output
python src_pretrain.py --dset a2p --gpu_id $gpu --home --output $output
python src_pretrain.py --dset a2r --gpu_id $gpu --home --output $output
python src_pretrain.py --dset a2c --gpu_id $gpu --home --output $output
python src_pretrain.py --dset r2a --gpu_id $gpu --home --output $output
python src_pretrain.py --dset r2p --gpu_id $gpu --home --output $output
python src_pretrain.py --dset r2c --gpu_id $gpu --home --output $output
python src_pretrain.py --dset c2r --gpu_id $gpu --home --output $output
python src_pretrain.py --dset c2a --gpu_id $gpu --home --output $output
python src_pretrain.py --dset c2p --gpu_id $gpu --home --output $output


python tar_adapt.py --home --dset a2r --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset r2a --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset r2c --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset r2p --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset p2a --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset p2c --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset a2p --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset a2c --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset p2r --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset c2a --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset c2p --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler
python tar_adapt.py --home --dset c2r --gpu_id $gpu --home --output $output --batch_size $batchsize --lr_encoder $scaler

