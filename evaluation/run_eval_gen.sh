python collect_gen_pc.py --src $1
python evaluate_gen_torch.py --src "${1}_pc" --n_test $3 -g $4
rm -rf "${1}_pc"

#Respectively:
#python collect_gen_pc.py --src ../proj_log/pretrained/lgan_1000/results/fake_z_ckpt200000_num9000_dec
#collect的--npoints参数决定了后面evaluate的点云形状，后面evaluate的--ntest不能小于前面npoints的值，否则会导致不一致。
#python evaluate_gen_torch.py --src ../proj_log/pretrained/lgan_1000/results/fake_z_ckpt200000_num9000_dec_pc --n_test 100 -g 0
#rm -rf ../proj_log/newDeepCAD/lgan_1000/results/fake_z_ckpt200000_num9000_dec_pc