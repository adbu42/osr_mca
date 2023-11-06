# Open Set Recognition using Masked Conditional Autoencoders
Improves the model from C2AE by replacing the convolutional UNet with Transfomers

Please edit the file config.yml according to your environment.  
Please add an api_key.yml with a "wandb_api_key" key to the directory.

Model for the UNet taken and adapted from https://github.com/milesial/Pytorch-UNet

Sources:
```
@inproceedings{oza2019c2ae,
  title={C2ae: Class conditioned auto-encoder for open-set recognition},
  author={Oza, Poojan and Patel, Vishal M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2307--2316},
  year={2019}
}

@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={16000--16009},
  year={2022}
}

@inproceedings{cai2022vision,
  title={A Vision Transformer Architecture for Open Set Recognition},
  author={Cai, Feiyang and Zhang, Zhenkai and Liu, Jie and Koutsoukos, Xenofon},
  booktitle={2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages={190--197},
  year={2022},
  organization={IEEE}
}

@inproceedings{perera2017extreme,
  title={Extreme value analysis for mobile active user authentication},
  author={Perera, Pramuditha and Patel, Vishal M},
  booktitle={2017 12th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2017)},
  pages={346--353},
  year={2017},
  organization={IEEE}
}
```
