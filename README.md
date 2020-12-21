# SATP_12.20

## 1. Annotation pipeline

1_annotation_pipeline-12.21.ipynb

## 2. Prepare our training/ testing data

2_Preprocess-Data-12.20.ipynb

## 3. Create word vector and dataloader from train.csv.
python create_input_files.py

## 4. Train our HAN model on train.csv
python train.py

## 5. Eval on our HAN model on test.csv
python eval.py

## 6. Visualiazation (Under developement)
python classify.py


## Results:

* Multi-label TEST ACCURACY 	 0.857

* RECALL SCORE		 0.526
* PRECISION SCORE	 0.769
* F1 SCORE			 0.625

* Relevant ACCURACY 	 0.885

* RECALL SCORE		 0.885
* PRECISION SCORE	 0.885
* F1 SCORE			 0.885
