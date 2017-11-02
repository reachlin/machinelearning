# machinelearning
machine learning algorithm sample code. Most of them are python. `tensorflow` folder includes implementation using tensorflow. 

## tensorflow/sample[1-9].py
Sample code to explain tensorflow concepts.

## tensorflow/sample_mnist.py
Basic linear classification using tensorflow. Although this is simple algorithm, but it explains major steps in machine learning very clearly.

## tensorflow/sample_cnn.py
A simple image classifier using CNN. It is modified to read black and white bitmap images containing a number in Chinese and outputs its arabic number, e.g. 七 as 7. The bitmap size must be 128 by 128.

```
docker run --name mlearn -v $PWD:/root/ -it gcr.io/tensorflow/tensorflow bash
cd /root/tensorflow
python sample_mnist.py --step 10000 # train 10k steps
python sample_mnist.py --predict data/0x4e03.bmp # predict an image
```

## mini-char-rnn.py
Modify [AK's RNN gist](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) to support unicode. It can also run on the history data of lottery to predict the next number:P

```
python mini-char-rnn.py -i input_chinese.txt -l 40
python mini-char-rnn.py -i lottery.txt -l 30 -n -test 08
```

Sample outputs of Chinese poems:

```
iter 10000, loss: 71.170911
----
 时 
赧郎长水月皆海。 
帝扫莫兮巴死水，寒高宿云鸯鞭间径，有尔同东黄金，栗藏访梦流，山女苍空乌去时，谢放营高晖。 
鸡为去浪淡被。 
落君如日。萧光芙馥州，秋君欲人相。 
郎花必地锦张鱼，飞之莫三云 
----
iter 11000, loss: 65.929996
iter 12000, loss: 61.579649
iter 13000, loss: 57.133800
iter 14000, loss: 53.556183
iter 15000, loss: 49.873506
iter 16000, loss: 47.075554
iter 17000, loss: 43.890813
iter 18000, loss: 41.661809
iter 19000, loss: 38.913145
iter 20000, loss: 36.891988
low loss, starting test now...
----
 青虎，何德横贵绣。 
六留君不独去，埋花数何骄。照师黩死萦支声。江酣落自望朝桃花上霜烟。 
黄鹤楼耳金玉，胡有月舟似，翡歌看低又入海，鼓水出几闲。系失垂杨江，转明霜珠。 
挥戈曲千愁欲行，功起楚清，然 
----
```

## tensorflow/sample_rnn.py
RNN with LSTM. TBC...

## Useful links
* [Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)
