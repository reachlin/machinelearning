# machinelearning
machinelearning stuff

## sample_cnn.py
A simple chinese number image classifier using CNN. It reads black and white bitmap images containing a number in Chinese and outputs its arabic number, e.g. 七 as 7. The bitmap size must be 128 by 128.

```
docker run --name mlearn -v $PWD:/root/ -it gcr.io/tensorflow/tensorflow bash
cd /root/tensorflow
python sample_mnist.py --step 10000 # train 10k steps
python sample_mnist.py --predict data/0x4e03.bmp # predict an image
```
