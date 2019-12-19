# 1.How to use Deep Taylor Decomposition 

## (1). Define model structure from trained model for calculation of relevance score

<pre><code>
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='scopename')
activations = tf.get_collection('activation_collection_name')
X = activations[0]

conv_ksize = "[convolution layerfilter size]"#[1, 4, 4 , 1]
pool_ksize = "[pooling layer filter size]"#[1 ,4, 4, 1]
conv_strides "[convolution layer stride size]"= #[1, 1, 1, 1]
pool_strides ="[pooling layer stride size]" #[1, 4, 4, 1]

weights.reverse()
activations.reverse()

taylor = Taylor(activations, weights, conv_ksize, pool_ksize, conv_strides, pool_strides, 'Taylor',part)

Rs = []
for i in range("number of class"):
    Rs.append(taylor(i))
</code></pre>

## (2).run session for getting relvance score
<pre><code>
model.sess.run([Rs],feed_dict={X:batch_in, model.keep_prob :p})
</code></pre>
ref : [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition]: https://arxiv.org/pdf/1512.02479.pd

* * *

# 2. How to use Network Dissection
## (1). please write down!


## Requirements 
+ tensorflow (1.9.0)
+ numpy (1.15.0)
+ matplotlib (2.2.2)

## License
[Apache License 2.0](https://github.com/OpenXAIProject/tutorials/blob/master/LICENSE "Apache")

## Contacts
If you have any question, please contact Ginkyeong Lee (gin908@unist.ac.kr), Sohee Cho (sohee.cho@kaist.ac.kr), Seonman Heo (smheo.cie@gmail.com )

<br /> 
<br />

# XAI Project 

**This work was supported by Institute for Information & Communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence (의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : KAIST, Korea Univ., Yonsei Univ., UNIST, AITRICS  

+ Web Site : <http://openXai.org>

