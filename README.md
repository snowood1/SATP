# SATP_12.20

#### 1. Annotation pipeline

1_annotation_pipeline-12.21.ipynb

#### 2. Prepare our training/ testing data

2_Preprocess-Data-12.20.ipynb

#### 3. Create word vector and dataloader from train.csv.
python create_input_files.py

#### 4. Train our HAN model on train.csv
python train.py

#### 5. Eval on our HAN model on test.csv
python eval.py

#### 6. Visualiazation (Under developement)
python classify.py


#### Results of HAN ( Unstable) :

  /| Accuracy  | Recall   | Precision | F1
---| ------------- | ------------- | ------------- | -------------
Multi-label  | 0.857  | 0.526  | 0.769 | 0.625
Relevant or Not  |  0.885  | 0.885  | 0.885  | 0.885
