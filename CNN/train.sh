python train.py --data_type '801'

python train.py --data_type '701' --rolling_weight_path 'resnet18_801_rew.pth.tar'




python train.py --data_type '801' --rolling_effect True --rolling_weight_path 'resnet18_901_rew.pth.tar' --dropouts False
python train.py --data_type '701' --pretrain_imagenet False --dropouts False --remove_pickle False
python train.py --data_type '701' --rolling_effect True --rolling_weight_path 'resnet18_801_rew.pth.tar' --dropouts False
python train.py --data_type '601' --pretrain_imagenet False --dropouts False --remove_pickle False
python train.py --data_type '601' --rolling_effect True --rolling_weight_path 'resnet18_701_rew.pth.tar' --dropouts False
python train.py --data_type '501' --pretrain_imagenet False --dropouts False --remove_pickle False
python train.py --data_type '501' --rolling_effect True --rolling_weight_path 'resnet18_601_rew.pth.tar' --dropouts False
python train.py --data_type '401' --pretrain_imagenet False --dropouts False --remove_pickle False
python train.py --data_type '401' --rolling_effect True --rolling_weight_path 'resnet18_501_rew.pth.tar' --dropouts False
python train.py --data_type '301' --pretrain_imagenet False --dropouts False --remove_pickle False
python train.py --data_type '301' --rolling_effect True --rolling_weight_path 'resnet18_401_rew.pth.tar' --dropouts False
python train.py --data_type '201' --pretrain_imagenet False --dropouts False --remove_pickle False
python train.py --data_type '201' --rolling_effect True --rolling_weight_path 'resnet18_301_rew.pth.tar' --dropouts False
