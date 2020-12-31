import time
import torch

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model import HierarchialAttentionNetwork
from utils import *
from datasets import HANDataset
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

# Data parameters
data_folder = './outdata'
word2vec_file = os.path.join(data_folder, 'word2vec_model')  # path to pre-trained word2vec embeddings
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)


classes = ['relevant', 'Armed Assault', 'Bombing/Explosion',  'Kidnapping', 'Other']

label_map = {k: v for v, k in enumerate(classes)}
rev_label_map = {v: k for k, v in label_map.items()}

# Model parameters
n_classes = len(label_map)
print(n_classes)
# word_rnn_size = 50  # word RNN size
# sentence_rnn_size = 50  # character RNN size
# word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
# sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
word_rnn_size = 200  # word RNN size
sentence_rnn_size = 200  # character RNN size
word_att_size = 400  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 400  # size of the sentence-level attention layer (also the size of the sentence context vector)




word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN

dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 64  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 4  # number of workers for loading data in the DataLoader
epochs = 7  # number of epochs to run
grad_clip = None  # clip gradients at this value
print_freq = 2000  # print training or validation status every __ batches
checkpoint = None  # path to model checkpoint, None if none

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

###  TODO eval

# Evaluation parameters
threshold = 0.5

####

def train(train_loader, model, criterion, optimizer, epoch,threshold = 0.5):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()

    # Batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):

        data_time.update(time.time() - start)

        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = 1.0 * labels.squeeze(1).to(device)  # (batch_size)

        # TODO ##################################
        # labels = labels[:,1:]

        # Forward prop.
        scores, word_alphas, sentence_alphas \
            = model(documents, sentences_per_document, words_per_sentence)
        # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)


        # Loss
        loss = criterion(scores, labels)  # scalar

        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()(scores, 1.0*labels)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update
        optimizer.step()

        # Find accuracy
        # _, predictions = scores.max(dim=1)  # (n_documents)
        # threshold = 0.5
        predictions = F.sigmoid(scores)
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        #
        # correct_predictions = torch.eq(predictions, labels).sum().item()
        # accuracy = correct_predictions / labels.size(0)
        accuracy =  accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy())

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs.update(accuracy, labels.size(0))

        start = time.time()

        # Print training status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  acc=accs))

def eval(test_loader, model,threshold):
    # Track metrics
    accs1 = AverageMeter()
    res1 = []
    lab1 = []

    accs2 = AverageMeter()
    res2 = []
    lab2 = []

    accs = AverageMeter()
    res = []
    lab = []

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
        y_multi = labels[:, 1:]  # True multi-labels
        y_relevant = labels[:, 0]  # True relevant labels

        pred = F.sigmoid(scores)
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        pred = pred.detach().cpu().numpy()  # predicted multi-labels

        # TODO:
        #  1. predicted relevant labels: if all the multi-labels are False, then the relevant labels will be False.
        #  2. Use relevant directly ( End to End with relevant)

        #  pred_relevant = list(map(int, pred_multi.any(axis=1)))
        pred_relevant = pred[:,0]
        pred_multi = pred[:,1:]


        # TODO : Multi-label classfication

        acc_multi = accuracy_score(y_multi, pred_multi)
        accs1.update(acc_multi, labels.shape[0])
        res1.extend(pred_multi.tolist())
        lab1.extend(y_multi.tolist())

        # TODO : is-relevant or not binary classfication

        # y_relevant =  labels[:,0]
        acc_relevant = accuracy_score(y_relevant, pred_relevant)
        accs2.update(acc_relevant, labels.shape[0])
        res2.extend(pred_relevant)
        lab2.extend(y_relevant)


        # Overall multi-label classification ( if we consider relevant as a class)
        acc = accuracy_score(labels, pred)
        accs.update(acc, labels.shape[0])
        res.extend(pred)
        lab.extend(labels)




    # Print final result

    # print('\n* Multi-label TEST ACCURACY \t %.3f' % (accs1.avg))
    # print("* RECALL SCORE\t\t %.3f" % recall_score(lab1, res1, average='micro'))
    # print("* PRECISION SCORE\t %.3f" % precision_score(lab1, res1, average='micro'))
    # print("* F1 SCORE\t\t\t %.3f" % f1_score(lab1, res1, average='micro'))
    #
    # # print(multilabel_confusion_matrix(lab1, res1))
    # # from sklearn.metrics import multilabel_confusion_matrix
    #
    # print('\n* Relevant ACCURACY \t %.3f' % (accs2.avg))
    # print("* RECALL SCORE\t\t %.3f" % recall_score(lab2, res2, average='micro'))
    # print("* PRECISION SCORE\t %.3f" % precision_score(lab2, res2, average='micro'))
    # print("* F1 SCORE\t\t\t %.3f" % f1_score(lab2, res2, average='micro'))
    # # print(multilabel_confusion_matrix(lab2, res2))
    #
    #
    # print('\n* Overall ACCURACY \t %.3f' % (accs.avg))
    # print("* RECALL SCORE\t\t %.3f" % recall_score(lab, res, average='micro'))
    # print("* PRECISION SCORE\t %.3f" % precision_score(lab, res, average='micro'))
    # print("* F1 SCORE\t\t\t %.3f" % f1_score(lab, res, average='micro'))

    df = pd.DataFrame(columns = ['threshold', 'type', 'accuracy',
        'recall(micro)', 'precision(micro)', 'f1(micro)',
        'recall(macro)', 'precision(macro)', 'f1(macro)',
        # 'recall(custom)', 'precision(custom)', 'f1(custom)',  'EMR', 'accuracy(custom)'
        'accuracy(custom)', 'precision(custom)','recall(custom)', 'f1(custom)',  'EMR']
    )

    df.loc[len(df)] = [threshold,
                        'multi-label',
                        accs1.avg,
                        recall_score(lab1, res1, average='micro'),
                        precision_score(lab1, res1, average='micro'),
                        f1_score(lab1, res1, average='micro'),

                       recall_score(lab1, res1, average='macro'),
                       precision_score(lab1, res1, average='macro'),
                       f1_score(lab1, res1, average='macro'),

                       accuracy_multilabel(lab1, res1),
                       precision_multilabel(lab1, res1),
                       recall_multilabel(lab1, res1),
                       f1_multilabel(lab1, res1),
                       accuracy_score(np.array(lab1), np.array(res1))
                        ]

    df.loc[len(df)] = [threshold,
                        'relevant',
                        accs2.avg,
                        recall_score(lab2, res2, average='micro'),
                        precision_score(lab2, res2, average='micro'),
                        f1_score(lab2, res2, average='micro'),

                       recall_score(lab2, res2, average='macro'),
                       precision_score(lab2, res2, average='macro'),
                       f1_score(lab2, res2, average='macro'),

                       None, None, None, None, None

                       # recall_multilabel(lab1, res1),
                       # precision_multilabel(lab1, res1),
                       # f1_multilabel(lab1, res1),
                       # accuracy_score(np.array(lab1), np.array(res1)),
                       # accuracy_multilabel(lab1, res1),
                        ]

    df.loc[len(df)] = [threshold,
                        'overall',
                        accs.avg,
                        recall_score(lab, res, average='micro'),
                        precision_score(lab, res, average='micro'),
                        f1_score(lab, res, average='micro'),

                       recall_score(lab, res, average='macro'),
                       precision_score(lab, res, average='macro'),
                       f1_score(lab, res, average='macro'),

                       accuracy_multilabel(lab, res),
                       precision_multilabel(lab, res),
                       recall_multilabel(lab, res),
                       f1_multilabel(lab, res),
                       accuracy_score(np.array(lab), np.array(res)),
                        ]

    return df

# def main():
"""
Training and validation.
"""
# global checkpoint, start_epoch, word_map


# Initialize model

embeddings, emb_size = load_word2vec_embeddings(word2vec_file, word_map)  # load pre-trained word2vec embeddings

model = HierarchialAttentionNetwork(n_classes=n_classes,
                                    vocab_size=len(word_map),
                                    emb_size=emb_size,
                                    word_rnn_size=word_rnn_size,
                                    sentence_rnn_size=sentence_rnn_size,
                                    word_rnn_layers=word_rnn_layers,
                                    sentence_rnn_layers=sentence_rnn_layers,
                                    word_att_size=word_att_size,
                                    sentence_att_size=sentence_att_size,
                                    dropout=dropout)

model.sentence_attention.word_attention.init_embeddings(embeddings)  # initialize embedding layer with pre-trained embeddings
model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)  # fine-tune

optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

criterion = nn.BCEWithLogitsLoss()

# Move to device
model = model.to(device)
criterion = criterion.to(device)


# DataLoaders
train_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'train'), batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'test'), batch_size=batch_size, shuffle=False,
                                          num_workers=workers, pin_memory=True)


# Epochs
for epoch in range(start_epoch, epochs):
    # One epoch's training
    print(epoch)
    train(train_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          epoch=epoch)

    # Decay learning rate every epoch
    # adjust_learning_rate(optimizer, 0.1)

    # Save checkpoint
    save_checkpoint(epoch, model, optimizer, word_map)

print('\n--- EVAL --- \n')

df = []
for threshold in np.linspace(0.1, 0.9, 9):
    print('\nthreshold', threshold)
    df.append(eval(test_loader, model, threshold))

df = pd.concat(df).reset_index(drop=True)

print(df[df.type == 'multi-label'].to_string())
print(df[df.type == 'relevant'].to_string())
print(df[df.type == 'overall'].to_string())

pd.set_option('display.max_columns', 30)

