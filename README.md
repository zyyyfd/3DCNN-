## Introduction
The CCMMOE model was constructed based on the MMOE model by introducing a redesigned expert
group and gating network. The reason for choosing the MMOE model is the significantly different 
adsorption energies and mechanisms observed for various adsorption molecules. 
The multi-expert gating mechanism of the MMOE model aims to enhance the model's capability to
learn different adsorption methods for various molecules. The CCMMOE model uses a voxelized adsorption 
model dataset as input features and an adsorption energy dataset as labels. 

This model has two outputs: adsorption energy (outputA) and the type of adsorbed molecule (outputB). 
Prior to designing the CCMMOE model, we trained a DNN model using perovskite feature dataset
and adsorption energy dataset as input features and labels, respectively. After analyzing the SHAP 
values obtained from the DNN model, it was discovered that the type of adsorbate molecule is the most
crucial feature influencing adsorption energy. Therefore, in CCMMOE, outputB is returned to gateA to 
improve the accuracy of adsorption energy prediction. 

The MAE values for outputA on the test and training 
sets were determined to be 4.3612 and 6.3082 kcal/mol, respectively. As for outputB, the accuracy
on the test and training sets were 94.8913% and 98.9068%, respectively. These high accuracies
indicate that the types of adsorbate molecules can be effectively distinguished, ensuring that gateA 
receives the correct molecules.
