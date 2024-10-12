python collect_gen_pc.py --src $1
python evaluate_gen_torch.py --src "${1}_pc" --n_test $2 -g $3
rm -rf "${1}_pc"

#Respectively:
#python collect_gen_pc.py --src ../proj_log/pretrained/lgan_1000/results/fake_z_ckpt200000_num9000_dec
#python evaluate_gen_torch.py --src ../proj_log/pretrained/lgan_1000/results/fake_z_ckpt200000_num9000_dec_pc --n_test 100 -g 0
#rm -rf ../proj_log/newDeepCAD/lgan_1000/results/fake_z_ckpt200000_num9000_dec_pc