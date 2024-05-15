# Reduced storage direct tensor ring decomposition for convolutional neural networks compression

## Data and baseline models
In experiments CIFAR-10 and ImageNet datasets are used. For CIFAR-10 datasets the pretrained models are downloaded from `torchvision` library. 
Models for CIFAR-10 dataset are trained from scratch.

## Requirements
* `pip install -r requirements`
* Download the ImageNet dataset from http://www.image-net.org/

## Training baseline models
After cloning the repository you can train baseline CIFAR-10 models by running the following command.
```shell
python cifar.py -net <net name> -mode <mode> -lr <lr> -epochs <epochs> -wd <wd> -b <b> -momentum <momentum>
```
where:
* `<net name>` - name of neural network, one from the following list:
  * `cifar10_resnet20`
  * `cifar10_resnet32`
  * `cifar10_resnet56`
  * `cifar10_vggnet`
* `<mode>` - there are two available modes, `train` for training and `fine_tune` for fine tuning
* `<lr>` - learning rate 
* `<epochs>` - number of epochs 
* `wd` - weight decay
* `<b>` - batch size
* `<momentum>` - value of momentum 

The following command shows the example of training ResNet-20 network from scratch.
```shell
python cifar.py -net cifar10_resnet20 -mode train -lr 0.1 -epochs 200 -wd 1e-4 -b 128 -momentum 0.9
```
  
## Compressing baseline models
For compression of baseline models use the following command:
```shell
python decompose_network.py -net <net name> -weights <weight path> -p <p>
```
where, 
* `<net name>` - name of neural network, one from the following list:
  * `cifar10_resnet20`
  * `cifar10_resnet32`
  * `cifar10_resnet56`
  * `cifar10_vggnet`
  * `imagenet_resnet18`
  * `imagenet_resnet34`
* `<weights>` - path of baseline network weights (only for CIFAR-10 models)
* `<p>`- prescribed relative error of tensor ring decomposition (float number in range `[0,1]`)

The following command shows the example of compressing ResNet-18 network.
```shell
python decompose_network.py -net imagenet_resnet18 -p 0.5
```

## Fine-tuning compressed network
For fine-tuning compressed network for CIFAR-10 dataset, use the following command:
```shell
python cifar.py -net <net name> -weights <weight path> -mode <mode> -lr <lr> -epochs <epochs> -wd <wd> -b <b> -momentum <momentum>
```
`<weight path>` is the path of compressed weights and `<mode>` has to be changed to `fine_tune`. The concrete example of fine tuning compressed ResNet-20 is shown below.
```shell
python cifar.py -net cifar10_resnet20 -weights decomposed_weights/tr_cifar10_resnet20_0_5.pth -mode fine_tune -lr 0.01 -epochs 160 -wd <wd> -b 128 -momentum 0
```

For fine-tuning compressed networks for ImageNet dataset used the following command.
```shell
python imagenet.py -weights <weight path> -lr <lr> -epochs <epochs> -wd <wd> -b <b> -momentum <momentum> -train <train path> -val <val path> -workers <workers> 
```
where
* `<train path>` - path of ImageNet training dataset
* `<val path>` - path of ImageNet validation dataset
* `<workers>` - number of workers

Example of fine tuning compressed ResNet-18 network is shown below.

For fine-tuning compressed networks for ImageNet dataset used the following command.
```shell
python imagenet.py -weights decomposed_weights/tr_cifar10_imagenet18_0_84.pth -lr 0.01 -epochs 30 -wd 0 -b 128 -momentum 0.9 -train train/ -val val/ -workers 4
```
  
