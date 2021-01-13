# SATP_12.20

#### 1. Annotation pipeline

1_annotation_parallel_2000_to_2020.ipynb


#### 2. Prepare our training/ testing data

2_Preprocess-Data-12.27.ipynb

#### 3. BiLSTM Baseline Deep Learning Model

3_BiLSTM.ipynb

#### 4. BERT

4_transformer_reduced_bert.ipynb

#### 5. Baselines: Traditional Machine learning models (SVM, LR) using 2-step pipeline.

5_Baselines_reduced-2-step.ipynb



#### 6. HAN

##### 6.1 Create word vector and dataloader from train.csv.
python create_input_files.py

##### 6.2 Train and eval HAN model
python HAN_end2end.py



#### Result:

|      | Accuracy | Precision | Recall    |  F1       | Exact Matching Ratio |
|   ---| -------- | --------- | --------- | ------    |------|
|One vs Rest    | 26.07	| 87.19	| 27.40	| 28.04	| 82.20 |
|Binary Relevance     | 26.65	| 86.08	| 28.29	| 28.40     | 82.47 |
|Class Chain     | 27.93	| 83.47	| 29.78	| 29.32	| 82.87 | 
|Label Powerset     | 28.86	| 85.53	| 30.51     | 30.05     | 83.14 |
|BiLSTM | 37.11 | 53.81 | 58.51 | 42.12 | 76.23 |
|HAN    | 52.98 | 65.64 | 74.82 | 55.78 | 83.07 |  
|BERT   | 62.52 | 74.73 | 80.01 | 64.59 | 87.23 | 
