from utils import create_input_files, train_word2vec_model

if __name__ == '__main__':
    create_input_files(csv_folder='./data',
                       output_folder='./outdata',
                       # sentence_limit=15,
                       # word_limit=20,
                       # min_word_count=5)
                       sentence_limit=30,
                       word_limit=200,
                       min_word_count=3,
                       label_columns = [1,2,3,4,5]  #       # 'news', 'is_relevant', 'Armed Assault', 'Bombing/Explosion', 'Kidnapping', 'Other'
                       )

    train_word2vec_model(data_folder='./outdata', algorithm='skipgram')
