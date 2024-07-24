gpu=0
batchsize=32
epoch=40
output="birds31_weight"

python src_pretrain.py --dset c2n --gpu_id $gpu --bird31 --output $output --batch_size $batchsize
python src_pretrain.py --dset n2c --gpu_id $gpu --bird31 --output $output --batch_size $batchsize
python src_pretrain.py --dset n2i --gpu_id $gpu --bird31 --output $output --batch_size $batchsize
python src_pretrain.py --dset i2n --gpu_id $gpu --bird31 --output $output --batch_size $batchsize
python src_pretrain.py --dset i2c --gpu_id $gpu --bird31 --output $output --batch_size $batchsize
python src_pretrain.py --dset c2i --gpu_id $gpu --bird31 --output $output --batch_size $batchsize

python tar_adapt.py --dset c2n --gpu_id $gpu --bird31 --output $output --batch_size $batchsize --max_epoch $epoch
python tar_adapt.py --dset n2c --gpu_id $gpu --bird31 --output $output --batch_size $batchsize --max_epoch $epoch
python tar_adapt.py --dset n2i --gpu_id $gpu --bird31 --output $output --batch_size $batchsize --max_epoch $epoch
python tar_adapt.py --dset i2n --gpu_id $gpu --bird31 --output $output --batch_size $batchsize --max_epoch $epoch
python tar_adapt.py --dset i2c --gpu_id $gpu --bird31 --output $output --batch_size $batchsize --max_epoch $epoch
python tar_adapt.py --dset c2i --gpu_id $gpu --bird31 --output $output --batch_size $batchsize --max_epoch $epoch
