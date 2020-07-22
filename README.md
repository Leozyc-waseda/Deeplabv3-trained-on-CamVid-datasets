# Deeplabv3-trained-on-CamVid-datasets
Deeplabv3+ trained on CamVid datasets, before train on your own dataset, it is just a practice .

# Environment 
Ubuntu16.04+cuda10.1+cuDNN7.4.2

tensorflow==1.15, tensorflow_gpu==1.15

GPU:NVIDIA 1070ti

Anaconda，用Anaconda好处是配置方便不容易崩。

检查是否成功？

anaconda-navigator #anaconda GUI编辑界面

nvidia‐settings #查看GPU信息

conda create ‐n tfgpu python=3.7 #创建tfgpu的虚拟名称

source activate tfgpu #启动名为tfgpu的虚拟环境

--------------------------------------------------------------------------------------
Ps:Solve kernel version dose not match DSO version problem

1.just sudo apt remove --purge nvidia*

2.sudo apt-get -f install #未能正常删除，系统提示命令，输入后问题解决

#如果nvidia正常删除，则参照Ubuntu重装GPU的博客

1.ctrl + alt +F1 

2.进入NVIDIA-xxxx.run文件的文件包，一般在Downloads下面

3.sudo service lightdm stop #停止x server，若是不关不能安装

4.sudo ./run file

5.NKMS choose no,would you like to run..? choose yes,install NVIDIA..OpenGL libraries? choose yes.
--------------------------------------------------------------------------------------


# Now,about how to train Camid dataset by use Deeplabv3+

1.git clone https://github.com/tensorflow/models.git #like export you can check deeplabv3+ github

2.test everything is ok or not.

cd ~/deeplab

python3 model_test.py #pip3 python3,do not use pip and python.use 3 you can solve a lot of problems.

if output ok,ok.

if no,see terminal,solve them.



# dataset processing

1.Download CamVid dataset

It should have [test,testannot,train,trainnnot,val,valannot] # annot means label

the number of train 367 ,test 233 ,val 101.

2.将所有图片放到两个文件夹下面

/home/(your user name)/dataset/CamVid/image: 存放所有的输入图片，共有701张，这其中包括训练集、测试集、 验证集的图片。


/home/(your user name)/dataset/CamVid/mask:存放所有的标签图片，共有701张，和image文件夹下的图片是一 一对应的。


对于CamVid数据集，创建了一个目录/home/bai/dataset/CamVid/index，该目录下包含三个.txt文 件:

train.txt:所有训练集的文件名称 

trainval.txt:所有验证集的文件名称

val.txt:所有测试集的文件名称



3.将image data 转换成TFRecord

1)mkdir tfrecord #tfrecord file

#将上述制作的数据集打包成TFRecord，使用的是build_voc2012_data.py

2)pwd :/home/(your user name)/models/research/deeplab/datasets

python build_voc2012_data.py \ 

--image_folder="/home/bai/dataset/CamVid/image" \ 

--semantic_segmentation_folder="/home/bai/dataset/CamVid/mask" \ 

--list_folder="/home/(your user name)/dataset/CamVid/index" \ 

--image_format="png" \ 

--output_dir="/home/(your user name)/dataset/CamVid/tfrecord"

image_folder :数据集中原输入数据的文件目录地址 

semantic_segmentation_folder:数据集中标签的文件目录地址 

list_folder : 将数据集分类成训练集、验证集等的指引目录文件目录 

image_format : 输入图片数据的格式，CamVid的是png格式 

#png 损失小

output_dir:制作的TFRecord存放的目录地址(自己创建)



# training part

修改训练脚本，在DeepLabv3+模型的基础上，主要需要修改以下两个文件， data_generator.py
train_utils.py 


1.添加数据集描述
在datasets/data_generator.py文件中，添加camvid数据集描述:

_CAMVID_INFORMATION = DatasetDescriptor(

    splits_to_sizes={
    
        'train': 367,  # num of samples in images/training
        
        'val': 101,  # num of samples in images/validation
        
    },
    
    num_classes=12, #camvid have 11 classes,add ignore_label = 12.
    
    ignore_label=255,
    
)

#注册数据集 同时在datasets/data_generator.py文件，添加对应数据集的名称:

_DATASETS_INFORMATION = {
'cityscapes': _CITYSCAPES_INFORMATION, 'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION, 'ade20k': _ADE20K_INFORMATION, 'camvid':_CAMVID_INFORMATION, #camvid示例 'mydata':_MYDATA_INFORMATION, #your own dataset
}


修改train_utils.py
对应的utils/train_utils.py中，关于 exclude_list 的设置修改，作用是在使用预训练权重时
候，不加载该 logit 层:

exclude_list = ['global_step','logits'] #add [,'logits']

if not initialize_last_layer:

exclude_list.extend(last_layers)



修改train.py
如果想在DeepLab的基础上fine-tune其他数据集， 可在deeplab/train.py中修改输入参数。 
其中有一些选项:
[使用预训练的所有权重]，设置initialize_last_layer=True 

[只使用网络的backbone]，设置initialize_last_layer=False和 

last_layers_contain_logits_only=False 使用所有的预训练权重，

除了logits以外。因为如果是[自己的数据集]，对应的classes不同(这个我 们前面已经设置不加载logits),

可设置initialize_last_layer=False和 last_layers_contain_logits_only=True




# Download pre-trained model ,because dataset is too small.

#xception_cityscapes_trainfine
https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md


#model path
/home/bai/models/research/deeplab/deeplabv3_cityscapes_train


# train command 

在目录 ~/models/research/deeplab下执行

python train.py \
--logtostderr \
--training_number_of_steps=300 \
--train_split="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \ 
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \ 
--train_crop_size=321,321 \
--train_batch_size=4 \
--dataset="camvid" \
--tf_initial_checkpoint='/home/(your user name)/models/research/deeplab/deeplabv3_cityscapes_train/model.ckpt' \
--train_logdir='/home/(your user name)/models/research/deeplab/exp/camvid_train/train' \
--dataset_dir='/home/(your user name)/dataset/CamVid/tfrecord'

#About train_crop_size , = output_stride * k + 1, where k is an integer. For example, we have 321x321，513x513 .



#测试结果可视化:
在目录 ~/models/research/deeplab下执行

python vis.py \
--logtostderr \
--vis_split="val" \
--model_variant="xception_65" \ 
--atrous_rates=6 \ 
--atrous_rates=12 \ 
--atrous_rates=18 \ 
--output_stride=16 \ 
--decoder_output_stride=4 \ 
--vis_crop_size=360,480 \
--dataset="camvid" \
--colormap_type="pascal" \ 
--checkpoint_dir='/home/(your user name)/models/research/deeplab/exp/camvid_train/train'\
--vis_logdir='/home/(your user name)/models/research/deeplab/exp/camvid_train/vis' \ 
--dataset_dir='/home/(your user name)/dataset/CamVid/tfrecord'

#vis_split:设置为测试集
vis_crop_size:设置360,480为图片的大小 dataset:设置为我们在data_generator.py文件设置的数据集名称 dataset_dir:设置为创建的TFRecord colormap_type:可视化标注的颜色 可到目录deeplab/exp/camvid_train/vis下查看可视化结果#



性能评估:

在目录 ~/models/research/deeplab下执行

python eval.py \ 
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=360,480 \
--dataset="camvid" \ 
--checkpoint_dir='/home/bai/models/research/deeplab/exp/camvid_train/train'\
--eval_logdir='/home/bai/models/research/deeplab/exp/camvid_train/eval' \ 
--dataset_dir='/home/bai/dataset/CamVid/tfrecord' \ 
--max_number_of_evaluations=1


#eval_split:设置为测试集 crop_size:同样设置为360和480 dataset:设置为camvid dataset_dir:设置为我们创建的数据集


查看mIoU值:
tensorboard --logdir /home/user/models/research/deeplab/exp/camvid_train/eval 查看训练过程的loss:

tensorboard --logdir /home/user/models/research/deeplab/exp/camvid_train/train


#PS:if didn't show anything

可以试试127.0.0.1:6006或者localhost:6006.

i will show the code later.








