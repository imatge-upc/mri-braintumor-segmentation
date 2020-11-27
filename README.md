# MRI Brain Tumor Segmentation and Uncertainty Estimation using 3D-Unet architectures on BraTS'20

This repository contains the code of the work presented in the paper [MRI Brain Tumor Segmentation and Uncertainty Estimation using 3D-Unet architectures]() 
which is used to participate on the BraTS'20 challenge on Brain Tumor Segmentation, for tasks 1 and 3. 

This work proposes the usage of V-Net and 3D-UNet based models for semantic segmentation in 3D-MRI Brain Tumor Segmentation and identifies certain and uncertain predictions at test time.

The original repository can be found [here](https://github.com/LauraMoraB/BrainTumorSegmentation).

## Repository Structure
    |__ resources/
        |__ config.ini
    |__ src/
        |__ dataset/
        |__ losses/
        |__ ensemble/
        |__ metrics/
        |__ post_processing/
        |__ test/
        |__ train/
        |__ uncertainty/
        |__ config.py
        |__ logging_conf.py
        |__ train.py
        |__ inference.py
        |__ normalize_uncertainty.py
        |__ run_post_processing.py
               
    |__ tests/
    |__ README.md

## Installation

```
pip install -r requirements.txt
```

## Execution

You can execute several processes, from the training of the model, inference, inference uncertainty, run post processing to the obtained results, compute metrics and compute an ensemble.

All scripts run similarly, as all the required configuration is read from the config.ini file. 
```
python <script.py> resources/config.ini
```

However, the `run_post_processing.py` is though to be run with SLURM arrays, so it will need editing in case you don't have a SLURM environment.

### Training

```
python train.py resources/config.ini
```

#### Network
Four Possible Networks:
* Basic VNet : vnet
* Deeper VNet: vnet_assym
* Basic 3DUNet: 3dunet
* Residual 3DUNet: 3dunet_residual

```ini
n_epochs: 100

init_features_maps: 32
network: 3dunet_residual or 3dunet or vnet_asymm or vnet

# unet based
unet_order: crg  
# cli  -  conv + LeakyReLU + instancenorm

# vnet asymm
non_linearity: relu
kernel_size: 3
padding: 1

# vnet
use_elu: true
```

#### Optimizer

Implemented: ADAM or SGD
```ini
optimizer: ADAM
learning_rate: 1e-4
weight_decay: 1e-5
# sgd only
momentum: 0.99
```

#### Loss

* Loss can be evaluated on ET/TC/WT (`eval_regions: true`) or ED/NCR/ET (`eval_regions: false`)
* Loss: dice, both_dice (dice eval + dice normal), gdl (not implemented with eval regions), combined (cross-entropy + dice)

```ini
loss: gdl
eval_regions: false
```

### Inference

Run as:
```
python inference.py resources/config.ini
```

#### Segmentation

```ini
[basics]
train_flag: false
compute_patches: false
resume: false

test_flag: true
uncertainty_flag: false
```

#### Uncertainty
3 Types of uncertainty can be computed: 
* aleatoric `uncertainty_type: tta` and `use_dropout: false`
* epistemic `uncertainty_type: ttd`
* both: `uncertainty_type: tta` and `use_dropout: true`

```ini
[basics]
train_flag: false
compute_patches: false
resume: false

test_flag: true
uncertainty_flag: true

[uncertainty]
n_iterations: 20
uncertainty_type: tta
use_dropout: false (used if uncertainty_type=tta)
```

## Model results 

### Task 1: Segmentation

| METHOD | DICE WT | DICE TC | DICE ET | HAUSDORFF WT | HAUSDORFF TC | HAUSDORFF ET|
| ------- | ----   | ---------| -------| ---------     | ---------   | ---------   |
| Basic V-Net                           | 0.8360	         | 0.7499           | 0.6159 	          | 26.4085          | 13.3398           | 49.7425 |      
| Basic V-Net + post                    | 0.8463             | 0.7526           | 0.6179              | 20.4073          | 12.1752           | 47.7020 |    
| Deeper V-Net                     	    | 0,8571	         | 0,7755	        | 0,6866	          | 16,0270	         | 17,6447	         | 44,0950|	       
| Deeper V-Net + post	                | 0,8611	         | 0,7790	        | 0,6897	          | 14,4988	         | 16,1533	         | 43,5184|	       
| Basic 3D-UNet                   	    | 0,8411	         | 0,7906	        | 0,6876	          | 13,3658	         | 13,6065	         | 50,9828|	     
| Basic 3D-UNet +post	                | 0,8052	         | 0,7749	        | 0,6742	          | 13,0969	         | 14,0047	         | 43,8928|	       
| Residual 3D-UNet	                    | 0,8072	         | 0,7740	        | 0,6955	          | 16,9635	         | 17,5142	         | 39,9172|	       
| Residual 3D-UNet + post         	    | 0,8142	         | 0,7748	        | 0,7119	          | 11,8505	         | 18,8146	         | 34,9652|
| Residual 3D-UNet-multiscale       	| 0,8172	         | 0,7664	        | 0,7071	          | 15,5342	         | 13,9380	         | 38,6098|	       
| Residual 3D-UNet-multiscale  + post	| 0,8246	         | 0,7647	        | 0,7163	          | 12,3372	         | 13,1045	         | 37,4224|	       
| Ensemble mean	                        | 0,8317	         | 0,7874	        | 0,6951	          | 13,4655	         | 12,9562	         | 47,5703|	       
| Ensemble mean + post	                | 0,8367	         | 0,7885	        | 0,7194              | 10,9320	         | 12,2427           | 37,9678|	     
| Ensemble majority	                     | 0,8223	         | 0,7801	        | 0,7003	          | 10,9781	         | 12,6571	         | 41,8566|	       
| Ensemble majority post	            | 0,8242	         | 0,7801	        | 0,7003	          | 10,0768          | 14,6322	         | 46,6045 |  



### Task 3: Uncertainty

| MEASURE  | METHOD | AUC DICE WT | AUC DICE TC |AUC DICE ET | FTP RATIO WT | FTP RATIO TC | FTP RATIO ET| FTN RATIO WT | FTM RATIO TC | FTN RATIO ET |
| ---------| -------| ---------   | ------------| ---------- | ---------    | ----------   | ---------   |  ---------   | ----------   | ---------   |
|Variance | TTA Residual 3D-UNet-multiscale       | 0,8316   | 0,7715	    | 0,7088	   | 0,0538	     | 0,0449       | 0,0380       | 0,0009	      | 0,0002  | 0,0001 |
|Variance | TTD Residual 3D-UNet-multiscale       | 0,8300	 | 0,7582	    | 0,7318	   | 0,1646	     | 0,1558	    | 0,0937	   | 0,0024	      | 0,0015	| 0,0004 |
|Variance | TTA + TTD Residual 3D-UNet-multiscale | 0,8325	 | 0,7632	    | 0,7276	   | 0,1812      | 0,1588	    | 0,0998	   | 0,0036       | 0,0020	| 0,0005 |
|Entropy  | TTA Residual 3D-UNet-multiscale       | 0,8326	 | 0,7816	    | 0,7138	   | 0,0635	     | 0,0476	    | 0,0362	   | 0,0011	      | 0,0047	|0,0063  |
|Entropy  | TTD Residual 3D-UNet-multiscale       | 0,8233	 | 0,7797       | 0,7423	   | 0,1512	     | 0,1285	    | 0,0698	   | 0,0021	      | 0,0082	| 0,0122 |
|Entropy  | TTA + TTD Residual 3D-UNet-multiscale |0,8343	 | 0,7909	    | 0,7710	   | 0,1525	     | 0,1213	    | 0,0664	   | 0,0030	      | 0,0101	| 0,0139 |