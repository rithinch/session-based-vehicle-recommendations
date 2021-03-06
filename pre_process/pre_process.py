import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='dataset file which has the click stream')
parser.add_argument('--dest', help='folder where to save the pre-processed data')
parser.add_argument('--split_days', type=int, default=3, help='folder where to save the pre-processed data')
opt = parser.parse_args()

dataset = opt.file
save_folder_name = opt.dest
overall_split_day = time.mktime(time.strptime("2019-03-15", '%Y-%m-%d'))

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    reader = csv.DictReader(f, delimiter=',')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = ''
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        
        curid = sessid
        
        item = data['reg_no'], int(data['timeframe'])

        curdate = data['date']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 1, curseq))
    if (len(filseq) < 2) or (sess_date[s] < overall_split_day):
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())



maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = maxdate - 86400 * opt.split_days

print('Splitting date', splitdate)

tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))
print(len(tes_sess))    
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(f"Number of unique cars: {item_ctr}")     
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = [] #Target/Ending node
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))


shrink_data = False

if shrink_data:
    split8 = int(len(tr_seqs) / 8)
    print(len(tr_seqs[-split8:]))

    tra8 = (tr_seqs[-split8:], tr_labs[-split8:])
    seq8 = tra_seqs[tr_ids[-split8]:]

    pickle.dump(tra, open('dataset_8/train.dat', 'wb'))
    pickle.dump(tes, open('dataset_8/test.dat', 'wb'))
    pickle.dump(tra_seqs, open('dataset_8/all_train_seq.dat', 'wb'))

def get_item_mappings(filename):
    clicks_df = pd.read_csv(filename)
    clicks_df.drop_duplicates(subset='reg_no', inplace=True)
    clicks_df.set_index('reg_no', inplace=True)
    clicks_df.drop(['client_id','session_id', 'date','time','timeframe'], axis=1, inplace=True)
    clicks_df['page'] = clicks_df['page'].apply(lambda x: f'https://shop.carstore.com{x}')
    d = clicks_df.to_dict('index')

    return dict((v,d[k]) for k,v in item_dict.items())

def get_item_features_embeddings(filename):

    clean_df = pd.read_csv(filename)
    clean_df.drop_duplicates(subset='reg_no', inplace=True)

    clean_df_c_len = len(clean_df.columns)

    make_one_hot_encoding = pd.get_dummies(clean_df['make'])
    clean_df = clean_df.drop('make', axis=1)
    clean_df = clean_df.join(make_one_hot_encoding)

    fuel_one_hot_encoding = pd.get_dummies(clean_df['fuel'])
    clean_df = clean_df.drop('fuel', axis=1)
    clean_df = clean_df.join(fuel_one_hot_encoding)

    t_one_hot_encoding = pd.get_dummies(clean_df['trasmission'])
    clean_df = clean_df.drop('trasmission', axis=1)
    clean_df = clean_df.join(t_one_hot_encoding)

    model_one_hot_encoding = pd.get_dummies(clean_df['model'])
    clean_df = clean_df.drop('model', axis=1)
    clean_df = clean_df.join(model_one_hot_encoding)

    features = clean_df.iloc[:,clean_df_c_len:]
    features = clean_df[['reg_no']].join(features)
    
    features.set_index('reg_no', inplace=True)

    d = features.to_dict('index')
    
    column_names = list(features.columns.values)

    features_dict = {}
    for k,v in item_dict.items():
        f = d[k]
        features_dict[v] = [f[c] for c in column_names]

    return features_dict





if not os.path.exists(save_folder_name):
    os.makedirs(save_folder_name)

pickle.dump(get_item_features_embeddings(dataset), open(f'{save_folder_name}/itemid_features.dat', 'wb'))
pickle.dump(item_dict, open(f'{save_folder_name}/reg_no_item_id.dat', 'wb'))
pickle.dump(get_item_mappings(dataset), open(f'{save_folder_name}/itemid_to_vehicle_mapping.dat', 'wb'))

pickle.dump(tra, open(f'{save_folder_name}/train.dat', 'wb'))
pickle.dump(tes, open(f'{save_folder_name}/test.dat', 'wb'))
pickle.dump(tra_seqs, open(f'{save_folder_name}/all_train_seq.dat', 'wb'))

print('Done.')

