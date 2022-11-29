python3 train.py \
--log_dir logs \
--exp_name stereonet_kitti \
--model StereoNet \
--check_val_every_n_epoch 5 \
--sync_batchnorm True \
--gpus -1 \
--max_steps 2000000 \
--accelerator cuda \
--max_disp 192 \
--optmizer RMS \
--lr 1e-3 \
--lr_decay 14000 0.9 \
--lr_decay_type Step \
--batch_size 8 \
--batch_size_val 8 \
--num_workers 2 \
--num_workers_val 2 \
--data_augmentation 1 \
--data_type_train KITTI2012 KITTI2015 \
--data_root_train /home/bz/Documents/SpatialAI/kitti_2012/training /home/bz/Documents/SpatialAI/kitti_2015/training \
--data_list_train lists/kitti2012_train170.list lists/kitti2015_train180.list \
--data_size_train 1152 320 \
--data_type_val KITTI2015 \
--data_root_val /home/bz/Documents/SpatialAI/kitti_2015/training \
--data_list_val lists/kitti2015_val20.list \
--data_size_val 1242 375 \
--pretrain ckpt/stereo_net.ckpt
