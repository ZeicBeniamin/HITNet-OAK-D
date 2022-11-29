python3 train.py \
--log_dir logs \
--exp_name hitnet_kitti \
--model HITNet_KITTI \
--gpus 1 \
--check_val_every_n_epoch 5 \
--max_steps 500000 \
--accelerator cuda \
--max_disp 256 \
--max_disp_val 192 \
--optmizer Adam \
--lr 4e-4 \
--lr_decay 400000 0.25 408000 0.1 410000 0.025 \
--lr_decay_type Lambda \
--batch_size 1 \
--batch_size_val 1 \
--num_workers 1 \
--num_workers_val 1 \
--data_augmentation 1 \
--data_type_train KITTI2012 KITTI2015 \
--data_root_train /home/bz/Documents/SpatialAI/kitti_2012/training /home/bz/Documents/SpatialAI/kitti_2015/training \
--data_list_train lists/kitti2012_train170.list lists/kitti2015_train180.list \
--data_size_train 1152 320 \
--data_type_val KITTI2015 \
--data_root_val /home/bz/Documents/SpatialAI/kitti_2015/training \
--data_list_val lists/kitti2015_val20.list \
--data_size_val 1242 375 \
--init_loss_k 3
