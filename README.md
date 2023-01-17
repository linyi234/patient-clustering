# Predicting Drug Treatment for Hospitalized Patients with Heart Failure

This repository houses the code used in our paper, [‘Predicting Drug Treatment for Hospitalized Patients with Heart Failure’](https://drive.google.com/file/d/1o6a6vGLz76YejaEqywJVzYF5UraLR7GH/view?usp=sharing). Linyi Zhou, Ioanna Miliou. PharML 2022: Machine Learning for Pharma and Healthcare Applications, 2022 (Workshop at ECML PKDD 2022). 

## Abstract

Heart failure and acute heart failure, the sudden onset or worsening of symptoms related to heart failure, are leading causes of hospital admission in the elderly. Treatment of heart failure is a complex problem that needs to consider a combination of factors such as clinical manifestation and comorbidities of the patient. Machine learning approaches exploiting patient data may potentially improve heart failure patients disease management. However, there is a lack of treatment prediction models for heart failure patients. Hence, in this study, we propose a workflow to stratify patients based on clinical features and predict the drug treatment for hospitalized patients with heart failure. Initially, we train the k-medoids and DBSCAN clustering methods on an extract from the MIMIC III dataset. Subsequently, we carry out a multi-label treatment prediction by assigning new patients to the pre-defined clusters. The empirical evaluation shows that k-medoids and DBSCAN successfully identify patient subgroups, with different treatments in each subgroup. DSBCAN outperforms k-medoids in patient stratification, yet the performance for treatment prediction is similar for both algorithms. Therefore, our work supports that clustering algorithms, specifically DBSCAN, have the potential to successfully perform patient profiling and predict individualized drug treatment for patients with heart failure. 

**Keywords:** *Drug treatment prediction* · *Heart failure* · *Acute heart failure* · *Machine Learning* · *Clustering*

## Dataset

All the tests have been performed using the MIMIC-III (v1.4) dataset (Johnson et al., 2016). Access to this dataset can be gained through [PhysioNet](https://physionet.org/content/mimiciii/1.4/). This repository does not contain any data from the dataset.

_____________________________________________________________________________________________
[Johnson, A. E. W., Pollard, T. J. & Mark, R. G. (2016), ‘Mimic-iii clinical database (version 1.4)’,
PhysioNet.](https://doi.org/10.13026/C2XW26 "Mimic-iii clinical database (version 1.4)")
