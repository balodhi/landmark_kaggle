python train.py --data_type '1001' --pretrain_imagenet False
python train.py --data_type '1001' --rolling_effect True
python train.py --data_type '901' --pretrain_imagenet False
python train.py --data_type '901' --rolling_effect True --rolling_weight_path 'resnet18_1001_rew.pth.tar'
python train.py --data_type '801' --pretrain_imagenet False
python train.py --data_type '801' --rolling_effect True --rolling_weight_path 'resnet18_901_rew.pth.tar'
python train.py --data_type '701' --pretrain_imagenet False
python train.py --data_type '701' --rolling_effect True --rolling_weight_path 'resnet18_801_rew.pth.tar'
python train.py --data_type '601' --pretrain_imagenet False
python train.py --data_type '601' --rolling_effect True --rolling_weight_path 'resnet18_701_rew.pth.tar'
python train.py --data_type '501' --pretrain_imagenet False
python train.py --data_type '501' --rolling_effect True --rolling_weight_path 'resnet18_601_rew.pth.tar'
python train.py --data_type '401' --pretrain_imagenet False
python train.py --data_type '401' --rolling_effect True --rolling_weight_path 'resnet18_501_rew.pth.tar'
python train.py --data_type '301' --pretrain_imagenet False
python train.py --data_type '301' --rolling_effect True --rolling_weight_path 'resnet18_401_rew.pth.tar'
python train.py --data_type '201' --pretrain_imagenet False
python train.py --data_type '201' --rolling_effect True --rolling_weight_path 'resnet18_301_rew.pth.tar'
python train.py --data_type '101' --pretrain_imagenet False
python train.py --data_type '101' --rolling_effect True --rolling_weight_path 'resnet18_201_rew.pth.tar'
