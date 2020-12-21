import time
from utils import *
from datasets import HANDataset
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
data_folder = './outdata/'

# Evaluation parameters
batch_size = 64  # batch size
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 2000  # print training or validation status every __ batches
checkpoint = 'checkpoint_han.pth.tar'

threshold = 0.5

# Load model
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Load test data
test_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'test'), batch_size=batch_size, shuffle=False,
                                          num_workers=workers, pin_memory=True)

# Track metrics
accs1 = AverageMeter()
res1 = []
lab1 = []

accs2 = AverageMeter()
res2 = []
lab2 = []

# Evaluate in batches
for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
        tqdm(test_loader, desc='Evaluating')):

    documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
    sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
    words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
    labels = labels.squeeze(1).to(device)  # (batch_size)

    # Forward prop.
    scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                 words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

    # Find accuracy
    labels = labels.detach().cpu().numpy()
    y_multi = labels[:,1:]  # True multi-labels
    y_relevant = labels[:,0] # True relevant labels

    pred_multi = scores.clone()
    pred_multi[pred_multi >= threshold] = 1
    pred_multi[pred_multi < threshold] = 0
    pred_multi = pred_multi.detach().cpu().numpy()  # predicted multi-labels

    # predicted relevant labels: if all the multi-labels are False, then the relevant labels will be False.
    pred_relevant = list(map(int, pred_multi.any(axis=1)))

    #TODO : Multi-label classfication



    acc_multi = accuracy_score(y_multi, pred_multi)
    accs1.update(acc_multi, labels.shape[0])
    res1.extend(pred_multi.tolist())
    lab1.extend(y_multi.tolist())


    #TODO : is-relevant or not binary classfication

    # y_relevant =  labels[:,0]
    acc_relevant = accuracy_score(y_relevant, pred_relevant)
    accs2.update(acc_relevant, labels.shape[0])
    res2.extend(pred_relevant)
    lab2.extend(y_relevant)



# Print final result

print('\n * Multi-label TEST ACCURACY \t %.3f\n' % (accs1.avg))
print("* RECALL SCORE\t\t %.3f" % recall_score(lab1, res1,  average='micro'))
print("* PRECISION SCORE\t %.3f" % precision_score(lab1, res1,  average='micro'))
print("* F1 SCORE\t\t\t %.3f" % f1_score(lab1, res1,  average='micro'))

print('\n * Relevant ACCURACY \t %.3f\n' % (accs2.avg))
print("* RECALL SCORE\t\t %.3f" % recall_score(lab2, res2,  average='micro'))
print("* PRECISION SCORE\t %.3f" % precision_score(lab2, res2,  average='micro'))
print("* F1 SCORE\t\t\t %.3f" % f1_score(lab2, res2,  average='micro'))
