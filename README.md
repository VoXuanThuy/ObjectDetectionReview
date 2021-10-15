## Anchor Assignment and Sampling Heuristics in Deep Object Detection: A Review
This repository provides a up-to-date paper list about anchor assigment, sampling heuristics and recent trends in object detection. This repository based on a problem-based taxonomy in the following paper: "Anchor Assignment and Sampling Heuristics in Deep Object Detection: A Review".
## How to add new papers to this repository
If you find a new paper that relates to anchor assignment, sampling methods as well as new trends in object detection. Please feel free to a pull request.
## News

## Table of Contents
1. [Anchor Assignment Methods](#1)  
    1.1 [Hard Anchor Assignment](#1.1)  
    1.2 [Soft Anchor Assignment](#1.2)     
2. [Sampling Methods](#2)  
    2.1 [Hard Sampling](#2.1)    
    2.2 [Soft Sampling](#2.2)  
3. [Recent Trends in Object Detection](#3)  
    3.1 [Transformer-based Detection Head](#3.1)  
    3.2 [Transformer-based Feature Extractor](#3.2)  


## 1. Anchor Assignment Methods <a name="1"></a>
#### 1.1. Hard Anchor Assignment <a name="1.1"></a>
  - Focal Loss for Dense Object Detection, ICCV 2017. [[Paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
  - FCOS: Fully Convolutional One-Stage Object Detection, ICCV 2019. [[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf)
  - Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection, CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.pdf)
  - You Only Look One-level Feature, CVPR 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_You_Only_Look_One-Level_Feature_CVPR_2021_paper.pdf)
#### 1.2. Soft Anchor Assignment <a name="1.2"></a>
  - FreeAnchor: Learning to Match Anchors for Visual Object Detection, NeurIPS 2019. [[Paper]](https://proceedings.neurips.cc/paper/2019/file/43ec517d68b6edd3015b3edc9a11367b-Paper.pdf)
  - Learning from Noisy Anchors for One-stage Object Detection, CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Learning_From_Noisy_Anchors_for_One-Stage_Object_Detection_CVPR_2020_paper.pdf)
  - Probabilistic Anchor Assignment with IoU Prediction for Object Detection, ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700358.pdf)
  - End-to-End Object Detection with Transformers, ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)
  - End-to-End Object Detection with Fully Convolutional Network, CVPR 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_End-to-End_Object_Detection_With_Fully_Convolutional_Network_CVPR_2021_paper.pdf)
  - LLA: Loss-aware label assignment for dense pedestrian detection, Neurocomputing 2021. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231221011796)
  - OTA: Optimal Transport Assignment for Object Detection, CVPR 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf)
  - What Makes for End-to-End Object Detection?, ICML 2021. [[Paper]](http://proceedings.mlr.press/v139/sun21b/sun21b.pdf)
  - YOLOX: Exceeding YOLO Series in 2021, arXiv 2021. [[Paper]](https://arxiv.org/pdf/2107.08430.pdf)
  - TOOD: Task-aligned One-stage Object Detection, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf)
## 2. Sampling Methods <a name="2"></a>
#### 2.1. Hard Sampling <a name="2.1"></a>
  - Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation, CVPR 2014. [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
  - Fast R-CNN, ICCV 2015. [[Paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
  - Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, NeurIPS 2015. [[Paper]](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)
  - SSD: Single Shot MultiBox Detector, ECCV 2016. [[Paper]](https://arxiv.org/pdf/1512.02325.pdf)
  - Training Region-based Object Detectors with Online Hard Example Mining, CVPR 2016. [[Paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)
  - Libra R-CNN: Towards Balanced Learning for Object Detection, CVPR 2019. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.pdf)
  - Overlap Sampler for Region-Based Object Detection, WACV 2020. [[Paper]](https://openaccess.thecvf.com/content_WACV_2020/papers/Chen_Overlap_Sampler_for_Region-Based_Object_Detection_WACV_2020_paper.pdf)
#### 2.2. Soft Sampling <a name="2.2"></a>
  - Focal Loss for Dense Object Detection, ICCV 2017. [[Paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
  - Gradient Harmonized Single-stage Detector, AAAI 2019. [[Paper]](https://arxiv.org/pdf/1811.05181.pdf)
  - Prime Sample Attention in Object Detection, CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Prime_Sample_Attention_in_Object_Detection_CVPR_2020_paper.pdf)
  - Equalization Loss for Long-Tailed Object Recognition, CVPR 2020. [[Paper]](https://arxiv.org/pdf/2003.05176.pdf)
  - Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection, CVPR 2021. [[Paper]](https://arxiv.org/pdf/2012.08548.pdf)
  - VarifocalNet: An IoU-aware Dense Object Detector, CVPR 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf)
## 3. Recent Trends in Object Detection <a name="3"></a>
#### 3.1. Transformer-based Detection Head <a name="3.1"></a>
  - End-to-End Object Detection with Transformers, ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)
  - Deformable DETR: Deformable Transformers for End-to-End Object Detection, ICML 2021. [[Paper]](https://openreview.net/forum?id=gZ9hCDWe6ke)
  - Fast Convergence of DETR with Spatially Modulated Co-Attention, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Fast_Convergence_of_DETR_With_Spatially_Modulated_Co-Attention_ICCV_2021_paper.pdf)
  - Conditional DETR for Fast Training Convergence, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Meng_Conditional_DETR_for_Fast_Training_Convergence_ICCV_2021_paper.pdf)
  - PnP-DETR: Towards Efficient Visual Analysis with Transformers, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_PnP-DETR_Towards_Efficient_Visual_Analysis_With_Transformers_ICCV_2021_paper.pdf)
  - Dynamic DETR: End-to-End Object Detection with Dynamic Attention, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Dai_Dynamic_DETR_End-to-End_Object_Detection_With_Dynamic_Attention_ICCV_2021_paper.pdf)
  - Rethinking Transformer-based Set Prediction for Object Detection, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper.pdf)
#### 3.2. Transformer-based Feature Extractor <a name="3.2"></a>
  - Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.pdf)
  - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)
  - Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding, ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Multi-Scale_Vision_Longformer_A_New_Vision_Transformer_for_High-Resolution_Image_ICCV_2021_paper.pdf)
  - Focal Self-attention for Local-Global Interactions in Vision Transformers, NeurIPS 2021. [[Paper]](https://arxiv.org/pdf/2107.00641.pdf)
  - CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows, arXiv 2021. [[Paper]](https://arxiv.org/pdf/2107.00652.pdf)

## Contact
If you have question, please contact Xuan-Thuy Vo, email: xthuy@islab.ulsan.ac.kr
