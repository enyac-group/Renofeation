# [Renofeation: A Simple Transfer Learning Method for Improved Adversarial Robustness](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Chin_Renofeation_A_Simple_Transfer_Learning_Method_for_Improved_Adversarial_Robustness_CVPRW_2021_paper.html)

This is the official repository for our paper [Renofeation: A Simple Transfer Learning Method for Improved Adversarial Robustness](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Chin_Renofeation_A_Simple_Transfer_Learning_Method_for_Improved_Adversarial_Robustness_CVPRW_2021_paper.html), which is presented at CVPR'21 Workshop on Fair, Data Efficient and Trusted Computer Vision and has received the **best paper award**.

## Don't fine-tune, Renofeate <img src="maintenance.png" width="30">

(Icon made by Eucalyp perfect from www.flaticon.com)

In our recent paper "[Renofeation: A Simple Transfer Learning Method for Improved Adversarial Robustness](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Chin_Renofeation_A_Simple_Transfer_Learning_Method_for_Improved_Adversarial_Robustness_CVPRW_2021_paper.html)", we show that numerous fine-tuning methods are vulnerable to [adversarial examples based on the pre-trained model](https://openreview.net/forum?id=BylVcTNtDS). This poses security concerns for indutrial applications that are based on fine-tuning such as [Google's Cloud AutoML](https://cloud.google.com/automl) and [Microsoft's Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/).

To combat such attacks, we propose _**Re**-training with **no**isy **fea**ture distilla**tion**_ or Renofeation. Renofeation does not start training from pre-trained weights but rather re-initialize the weights and train with noisy feature distillation. To instantiate noisy feature distillation, we incorporate [spatial dropout](https://arxiv.org/abs/1411.4280) and [stochastic weight averaging](https://arxiv.org/abs/1803.05407) with feature distillation to avoid over-fitting to the pre-trained feature without hurting the generalization performance, which in turn improves the robustness.

To this end, we demonstrate empirically that the attack success rate can be reduced from 74.3%, 65.7%, 70.3%, and 50.75% down to 6.9%, 4.4%, 4.9%, and 6.9% for ResNet-18, ResNet-50, ResNet-101, and MobileNetV2, respectively. Moreover, the clean-data performance is comparable to the fine-tuning baseline!

For more details and an ablation study of our proposed method, please check out our [paper]()!


## Dependency

- PyTorch 1.0.0
- TorchVision 0.4.0
- AdverTorch 0.2.0

## Preparing datasets

### [Caltech-UCSD 200 Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html)
Layout should be the following for the dataloader to load correctly

```
CUB_200_2011/
|    README
|    bounding_boxes.txt
|    classes.txt
|    image_class_labels.txt
|    images.txt
|    train_test_split.txt
|--- attributes
|--- images/
|--- parts/
|--- train/
|--- test/
```

### [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
```
Flower_102/
|    imagelabels.mat
|    setid.mat
|--- jpg/
```

### [Stanford 120 Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
```
stanford_dog/
|    file_list.mat
|    test_list.mat
|    train_list.mat
|--- train/
|--- test/
|--- Images/
|--- Annotation/
```

### [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html)
```
stanford_40/
|    attributes.txt
|--- ImageSplits/
|--- JPEGImages/
|--- MatlabAnnotations/
|--- XMLAnnotations/
```

### [MIT 67 Indoor Scenes](http://web.mit.edu/torralba/www/indoor.html)
```
MIT_67/
|    TrainImages.txt
|    TestImages.txt
|--- Annotations/
|--- Images/
|--- test/
|--- train/
```

## Model training and evaluation

**Be sure to modify the data path for each datasets for successful runs**

The scripts for training and evaluating *DELTA*, *Renofeation*, and *Re-training* are under `scripts/`.


## Citation

```
@InProceedings{Chin_2021_CVPR,
    author    = {Chin, Ting-Wu and Zhang, Cha and Marculescu, Diana},
    title     = {Renofeation: A Simple Transfer Learning Method for Improved Adversarial Robustness},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {3243-3252}
}
```