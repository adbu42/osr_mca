# Open Set Recognition using Contrastive Conditional Autoencoders
Traditional machine learning models assume that only images from training data classes, called the closed set, appear in the prediction task. 
In real-world applications unknown images from outside these classes, called open-set images, can pollute the prediction task, leading to them being wrongly classified as closed-set images. 
Open-set recognition models address this problem by splitting the prediction task into closed-set classification and differentiation between open- and closed-set images. One State of the Art Open Set Recognition model is C2AE, short for "Class Conditioned Auto-Encoders". 
This model uses an auto-encoder to reconstruct input images only when they correspond to one of the closed set classes. Badly reconstructed images are labeled as open-set, while closed-set images are classified with the encoder part of the auto-encoder.
This work improves C2AE in two ways. Firstly auto-encoder architectures with different encoder and decoder depths are compared on open and closed set performance. 
Tests on number and object classification datasets show that a combination of an adapted Wide Resnet encoder and a shallow decoder with skip connections akin to a U-Net outperforms all other architecture approaches. 
This auto-encoder also outperforms a simple baseline only using the SoftMax output from the encoder part. Secondly contrastive pretraining is applied to encoder weights before the start of C2AEs training loop. 
Contrastive learning lets the encoder learn useful image features by mapping similar instances closer together in the representation space while pushing dissimilar instances farther apart. Both supervised and self-supervised contrastive pretraining are used. 
Experiments show that supervised pretraining achieves the highest open-set and closed-set results, beating both self-supervised and the original framework without pretraining.

Please edit the file config.yml according to your environment.  
Please add an api_key.yml with a "wandb_api_key" key to the "main_model" directory.

Sources:
```
@inproceedings{c2ae,
  title={C2ae: Class conditioned auto-encoder for open-set recognition},
  author={Oza, Poojan and Patel, Vishal M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2307--2316},
  year={2019}
}

@inproceedings{resnet,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{u_net,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18},
  pages={234--241},
  year={2015},
  organization={Springer}
}


@book{evt,
  title={An introduction to statistical modeling of extreme values},
  author={Coles, Stuart and Bawa, Joanna and Trenner, Lesley and Dorazio, Pat},
  volume={208},
  year={2001},
  publisher={Springer}
}

@article{widenet,
  title={Wide residual networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1605.07146},
  year={2016}
}

@inproceedings{contrastive_self_sup,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International conference on machine learning},
  pages={1597--1607},
  year={2020},
  organization={PMLR}
}

@article{contrastive_sup,
  title={Supervised contrastive learning},
  author={Khosla, Prannay and Teterwak, Piotr and Wang, Chen and Sarna, Aaron and Tian, Yonglong and Isola, Phillip and Maschinot, Aaron and Liu, Ce and Krishnan, Dilip},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={18661--18673},
  year={2020}
}

@inproceedings{autoaugment,
  title={Autoaugment: Learning augmentation strategies from data},
  author={Cubuk, Ekin D and Zoph, Barret and Mane, Dandelion and Vasudevan, Vijay and Le, Quoc V},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={113--123},
  year={2019}
}
```
