Curls & Whey: Boosting Black-Box Adversarial Attacks (accepted to CVPR2019)
============================================================

Introduction
------------------------------------------------------------
Image classifiers based on deep neural networks suffer from harassment caused by adversarial examples. Two defects exist in black-box iterative attacks that generate adversarial examples by incrementally adjusting the noise-adding direction for each step. On the one hand, existing iterative attacks add noises monotonically along the direction of gradient ascent, resulting in a lack of diversity and adaptability of the generated iterative trajectories. On the other hand, it is trivial to perform adversarial attack by adding excessive noises, but currently there is no refinement mechanism to squeeze redundant noises. In this work, we propose Curls & Whey black-box attack to fix the above two defects. During Curls iteration, by combining gradient ascent and descent, we ‘curl’ up iterative trajectories to integrate more diversity and transferability into adversarial examples. Curls iteration also alleviates the diminishing marginal effect in existing iterative attacks. The Whey optimization further squeezes the ‘whey’ of noises by exploiting the robustness of adversarial perturbation. Extensive experiments on Imagenet and Tiny-Imagenet demonstrate that our approach achieves impressive decrease on noise magnitude in l2 norm. Curls & Whey attack also shows promising transferability against ensemble models as well as adversarially trained models. In addition, we extend our attack to the targeted misclassification, effectively reducing the difficulty of targeted attacks under black-box condition.
![](https://github.com/walegahaha/Curls-Whey/raw/master/figures/curls_whey.png)


Paper
------------------------------------------------------------
Yucheng Shi, Siyu Wang, Yahong Han. "Curls & Whey: Boosting Black-Box Adversarial Attacks." CVPR 2019 (Oral).

Reference
------------------------------------------------------------
If you find this useful in your work, please consider citing the following reference:
```
@inproceedings{CurlsWhey2019CVPR,
    title = {Curls & Whey: Boosting Black-Box Adversarial Attacks},
    author = {Shi, Yucheng and Wang, Siyu and Han, Yahong},
    booktitle = {Computer Vision and Pattern Recognition (CVPR), 2019},
    year = {2019}
}
```

Datasets
------------------------------------------------------------
The datasets used in the paper are available at the following links:
* [Imagenet](http://image-net.org/index)
* [Tiny Imagenet](https://tiny-imagenet.herokuapp.com/)

Environment
------------------------------------------------------------
The code is developed using python 3.5 and pytorch 0.4.1 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 1 GeForce GTX TITAN X GPU cards. 

Usage
------------------------------------------------------------
./bmodels/inceptionv3/inceptionv3.pt&ensp;&ensp;&ensp;&ensp;(https://pan.baidu.com/s/1_j7gVcGcWaobgJi7K11e6A)&ensp;&ensp;&ensp;&ensp;&ensp; code: tcax <Br/>
./fmodels/resnet/resnet101.pt&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;(https://pan.baidu.com/s/19kQBVwhtZw4mgHuarwFQjQ)&ensp;&ensp; code: z2w9 <Br/>
./temp.zip&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;(https://pan.baidu.com/s/1CMvpGyGKwFpKV1lPhr4FUA)&ensp;&ensp;&ensp; code: qk6y <Br/>
<Br/>
unzip temp.zip <Br/>
pip --no-cache-dir install  -r requirements.txt <Br/>
<Br/>
python untargeted_attack.py <Br/>
python targeted_attack.py

Examples
------------------------------------------------------------
![](https://github.com/walegahaha/Curls-Whey/raw/master/figures/example_figure_untargeted.png)  
<br/> 
![](https://github.com/walegahaha/Curls-Whey/raw/master/figures/example_figure_targeted.png)
