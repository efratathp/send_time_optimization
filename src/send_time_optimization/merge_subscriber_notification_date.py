import getopt
import os
import pathlib
import re
import sys
import numpy as np
import pandas as pd
import datetime

# ===========================
#       Consts
# ===========================
INCLUDE_USERS_WO_CLICK_HISTORY = False
SPLIT_TRAIN_TEST_BY_DOY = 109
KEEP_WEEKDAY_ONLY = True
CNVRT_AGG_HR_TO_DIST = True
PREDICT_BY_DIST = True
TEST_PREDICT_DIST = PREDICT_BY_DIST
FIRST_TEST_DOY = 109
LAST_TEST_DOY = 112
SEED = 1123456
RUN_RAND = 0
PREDICT_BY_SUBSCRIPTION_HOUR = 1
HOUR_DELTA = 0
# ===========================
#       Input
# ===========================
domain_id = 1274926143 #[ 308132225, 1274926143] joy,western
fn_domain_notifications_date = 'data/relevant_notification_90days_202104281022.csv'
fn_domain_subscribers = 'data/all_subscribers_of_2domains_202105052203.csv'
if domain_id == 308132225:
    fn_domain_subscribers_click = 'data/joy_land_subscribers_notifications'
if domain_id == 1274926143:
    fn_domain_subscribers_click = 'data/western_subscribers_notifications'

# ===========================
#       Input validation
# ===========================
assert(pathlib.Path(fn_domain_notifications_date).exists())
assert(pathlib.Path(fn_domain_subscribers).exists())
assert(pathlib.Path(fn_domain_subscribers_click).exists())


# ===========================
#       Data processing
# ===========================
# ====================== SPLIT Train - Test ==========================
def split_by_day_of_year(k1k2_n_s_c, day_of_year_to_split_by):
    df_test = k1k2_n_s_c[k1k2_n_s_c.ts_dayofyear >= day_of_year_to_split_by].copy()
    df_train = k1k2_n_s_c[k1k2_n_s_c.ts_dayofyear < day_of_year_to_split_by].copy()

    return df_test, df_train #, df_train_label

def create_label_columns_doy_hr(df_test):
    # REFACTORed so can be run on the df_test without weekends
    df_train_label = pd.pivot_table(df_test[['subscriber_id', 'ts_dayofyear', 'ts_hour', 'is_open']], index='subscriber_id', columns=['ts_dayofyear', 'ts_hour'], values=['is_open'], aggfunc=np.sum) #TODO consider adding , fill_value=0
    col_names = [('label_{}_{}_{}'.format(col[0], int(col[1]), col[2]).strip('.0')) for col in df_train_label.columns.values]
    df_train_label.columns = col_names
    df_train_label.reset_index(inplace=True)

    return df_train_label #df_test, df_train, df_train_label

def create_label_columns_hr(df_test):
    # REFACTORed so can be run on the df_test without weekends
    df_train_label = pd.pivot_table(df_test[['subscriber_id', 'ts_hour', 'is_open']], index='subscriber_id', columns=['ts_hour'], values=['is_open'], aggfunc=np.sum) #TODO consider adding , fill_value=0
    col_names = [('label_{}_{}'.format(col[0], int(col[1])).strip('.0')) for col in df_train_label.columns.values]
    df_train_label.columns = col_names
    df_train_label.reset_index(inplace=True)

    return df_train_label #df_test, df_train, df_train_label

# ====================== Aggs by dow & hr ==========================
def agg_df_by_subscriber_dow_hr(k1k2_n_s_c):
    #df2agg = k1k2_n_s_c[(k1k2_n_s_c.min_created_at<=k1k2_n_s_c.send_at)].copy()
    pv = pd.pivot_table(k1k2_n_s_c[['subscriber_id', 'is_open', 'is_notify', 'ts_hour', 'ts_dayofweek']],
                        index=['subscriber_id','ts_dayofweek'],
                        columns=['ts_hour'],
                        values=['is_open', 'is_notify'], fill_value=0, aggfunc=np.sum)

    # Flatten col names #TODO change the strip to different round of time either every 1 hour or ever 15min
    col_names = [('{}_{}'.format(col[0], col[1]).strip('.0')) for col in pv.columns.values]
    pv.columns = col_names
    pv.reset_index(inplace=True)

    #TODO create features of Per_hour(#opens/#notifies), Per_hour(#opens/#total_daily_opens)
    # TODO 1-hot-encoder for dow
    #TODO predict func
    return pv

# ====================== Aggs only by hour ==========================
def agg_df_by_subscriber_hr(k1k2_n_s_c):
    pv = pd.pivot_table(k1k2_n_s_c[['subscriber_id', 'is_open', 'is_notify', 'ts_hour']],
                        index=['subscriber_id'],
                        columns=['ts_hour'],
                        values=['is_open', 'is_notify'], fill_value=0, aggfunc=np.sum)

    # Flatten col names #TODO change the strip to different round of time either every 1 hour or ever 15min
    col_names = [('{}_{}'.format(col[0], col[1]).strip('.0')) for col in pv.columns.values]
    pv.columns = col_names
    pv.reset_index(inplace=True)

    #TODO create features of Per_hour(#opens/#notifies), Per_hour(#opens/#total_daily_opens)
    # TODO 1-hot-encoder for dow


    return pv


# ====================== keep week days ==========================
def keep_week_day_only(df):
    return df[df.ts_dayofweek.isin([1, 2, 3, 4])]


# ====================== MODELING ==========================
def cnvrt_is_open_hr_agg_to_dist(df): #df_test
    # input : each subscriber is aggregated by hour
    # output : subscriber
    is_open_cols_sub = [col for col in df.columns if ('is_open_' in str(col).lower())]
    is_notify_cols = [col for col in df.columns if ('is_notify_' in str(col).lower())]
    is_open_cols = is_open_cols_sub.copy()
    is_open_cols_sub.append('subscriber_id')
    df_is_open = df[is_open_cols_sub].copy()
    df_is_open['is_open_sum'] = 0
    #df_is_open.loc[:, 'is_open_sum'] = 0
    df_is_open_dist=df.copy()
    df_is_open_dist['is_open_sum'] = df_is_open[is_open_cols].sum(axis=1)
    df_is_open['is_open_sum'] = df_is_open[is_open_cols].sum(axis=1)
    # added 1 to avoid division by 0
    df_is_open.loc[df_is_open['is_open_sum'] == 0, 'is_open_sum'] = 1

    for i in is_open_cols:
        df_is_open_dist[i] = df_is_open[i]  / (df_is_open['is_open_sum'] )
        df_is_open_dist[i.replace('is_open', 'is_open_notify')] = df_is_open[i] / (df[i.replace('is_open', 'is_notify')] + 1)

    return df_is_open_dist


def predict(df_dist_w_label, hr):
    is_open_cols_sub = [col for col in df_dist_w_label.columns if (('is_open_' in str(col).lower()) and not(col=='is_open_sum') and not(('is_open_notify' in str(col).lower())))]
    is_open_cols = is_open_cols_sub.copy()
    is_open_cols_sub.append('subscriber_id')
    df_dist = df_dist_w_label[is_open_cols].copy()
    df_dist_w_label['pred'] = df_dist.idxmax(axis=1)  # todo return the df_dist_w_label  with a column of the prediction
    ls_subscriber_at_ts_hr = df_dist_w_label[df_dist_w_label['pred']=='is_open_'+str(hr)].subscriber_id
    ls_relevant_subscriber = df_dist_w_label[df_dist_w_label.is_open_sum > 0].subscriber_id
    return ls_subscriber_at_ts_hr, df_dist_w_label, ls_relevant_subscriber

def predict_by_subscription_hour(df_dist_w_label, hr, doy=None, dow=None):
    is_open_cols_sub = [col for col in df_dist_w_label.columns if (('is_open_' in str(col).lower()) and not(col=='is_open_sum') and not(('is_open_notify' in str(col).lower())))]
    is_open_cols = is_open_cols_sub.copy()
    is_open_cols_sub.append('subscriber_id')
    df_dist = df_dist_w_label[is_open_cols].copy()
    df_cp = df_dist_w_label.copy()

    # todo return the df_dist_w_label  with a column of the prediction
    if doy :
        df_cp['pred_doy'] = doy
    if dow :
        df_cp['pred_dow'] = dow
    df_cp['pred_hr'] = hr

    # return the hour of subscription

    pred_hrs = df_cp['min_created_at_hr'].apply(lambda h: 'is_open_' + str(int(h+HOUR_DELTA)))

    df_cp.loc[:,'pred']=pred_hrs.loc[:]



    #candidates same size as distribution
    #distribution - must be array of normalized values not function
    #size is number of values to return
    #replace is sampling with replacement

    ls_subscriber_at_ts_hr = df_cp[df_cp['pred']=='is_open_'+str(hr)].subscriber_id
    ls_relevant_subscriber = df_dist_w_label[df_dist_w_label.is_open_sum > 0].subscriber_id
    return ls_subscriber_at_ts_hr, df_cp, ls_relevant_subscriber


def predict_rand(df_dist_w_label, hr, doy=None, dow=None):
    is_open_cols_sub = [col for col in df_dist_w_label.columns if (('is_open_' in str(col).lower()) and not(col=='is_open_sum') and not(('is_open_notify' in str(col).lower())))]
    is_open_cols = is_open_cols_sub.copy()
    is_open_cols_sub.append('subscriber_id')
    df_dist = df_dist_w_label[is_open_cols].copy()
    df_cp = df_dist_w_label.copy()

    # todo return the df_dist_w_label  with a column of the prediction
    if doy :
        df_cp['pred_doy'] = doy
    if dow :
        df_cp['pred_dow'] = dow
    df_cp['pred_hr'] = hr

    #df_dist_w_label['pred'] = df_dist.idxmax(axis=1)
    # returns ndarray np
    rng = np.random.default_rng(seed=SEED)
    N = 1
    candidates = is_open_cols.copy()
    distribution = np.random.uniform(low=0.0, high=1.0, size=len(candidates))
    p = distribution/distribution.sum()
    p = None
    # works over matrix where each row is a probability
    # xx=df_cp.loc[:1,is_open_cols].copy()
    # xx.apply(lambda p: rng.choice(a=is_open_cols, p=np.array([x for x in p])/p.sum(), size=min(N, len(p)), replace=True), axis=1)
    # pred_hrs = df_cp[is_open_cols].apply(lambda p: rng.choice(a=is_open_cols, p=None, size=min(N, len(p)), replace=True), axis=1)
    # df_cp.loc[:,'pred']=pred_hrs.loc[:]
    if df_cp[is_open_cols].sum().sum():

        #pred_hrs = df_cp[is_open_cols].apply(lambda p: rng.choice(a=is_open_cols, p=np.array([x for x in p])/p.sum(), size=min(N, len(p)), replace=True), axis=1)
        # Handling rows with 0 distribution
        # value_when_true if condition else value_when_false
        pred_hrs = df_cp[is_open_cols].apply(
            lambda p: rng.choice(a=is_open_cols, p=(None if p.sum()==0 else np.array([x for x in p]) / p.sum()), size=min(N, len(p)),
                                 replace=True), axis=1)

        df_cp.loc[:,'pred']=pred_hrs.loc[:]
    else:
        pred_hrs = rng.choice(a=candidates, p=None, size=max(N, df_cp.shape[0]), replace=True)
        df_cp.loc[:, 'pred'] = pred_hrs[:]



    #candidates same size as distribution
    #distribution - must be array of normalized values not function
    #size is number of values to return
    #replace is sampling with replacement

    ls_subscriber_at_ts_hr = df_cp[df_cp['pred']=='is_open_'+str(hr)].subscriber_id
    ls_relevant_subscriber = df_dist_w_label[df_dist_w_label.is_open_sum > 0].subscriber_id
    return ls_subscriber_at_ts_hr, df_cp, ls_relevant_subscriber

def test_predict(df_pred_ls, df_label, ls_relevant_subscriber):
    df = df_label[df_label.subscriber_id.isin(ls_relevant_subscriber)].copy() #TODO add columns 1 for predicted and for relevant
    df_nr = df_label[df_label.subscriber_id.isin(ls_relevant_subscriber)==False].copy() #TODO add columns 1 for predicted and for relevant

    print( 'TP: ' , np.sum(df_label[df_label.subscriber_id.isin(df_pred_ls)][df_label.columns[1]]>0))
    pred_prec_a = np.sum(df_label[df_label.subscriber_id.isin(df_pred_ls)][df_label.columns[1]] > 0) / df_pred_ls.shape[0]
    true_prec_a = np.sum(df_label[df_label.columns[1]] > 0) / df_label.shape[0]

    col_name_pred = df_label.columns[1]
    df_label_cp = df_label.copy()
    df_label_cp['is_relevant'] = 0
    df_label_cp.loc[df_label_cp.subscriber_id.isin(ls_relevant_subscriber),'is_relevant'] = 1
    df_label_cp['pred_'+ col_name_pred] = 0
    df_label_cp.loc[df_label_cp.subscriber_id.isin(df_pred_ls), 'pred_'+ col_name_pred] = 1

    tp_a = np.sum(df_label[df_label.subscriber_id.isin(df_pred_ls)][df_label.columns[1]]>0)
    fp_a = np.sum(df_label[df_label.subscriber_id.isin(df_pred_ls)][df_label.columns[1]]==0)
    tn_a = np.sum(df_label[df_label.subscriber_id.isin(df_pred_ls)==False][df_label.columns[1]]==0)
    fn_a = np.sum(df_label[df_label.subscriber_id.isin(df_pred_ls)==False][df_label.columns[1]]>0)
    pred_prec = np.sum(df[df.subscriber_id.isin(df_pred_ls)][df.columns[1]]>0) / df_pred_ls.shape[0]
    true_prec = np.sum(df[df.columns[1]]>0) / df.shape[0]
    tp = np.sum(df[df.subscriber_id.isin(df_pred_ls)][df.columns[1]] > 0)
    fp = np.sum(df[df.subscriber_id.isin(df_pred_ls)][df.columns[1]] == 0)
    tn = np.sum(df[df.subscriber_id.isin(df_pred_ls) == False][df.columns[1]] == 0)
    fn = np.sum(df[df.subscriber_id.isin(df_pred_ls) == False][df.columns[1]] > 0)

    recall_pred = tp/(tp+fn)
    recall_pred_a = tp_a/(tp_a+fn_a)
    prec = tp / (tp + fp)

    F1 =  2 * prec * recall_pred/(prec + recall_pred)# 2*(precision * recall)/(precision + recall)

    tp_nr = np.sum(df_nr[df_nr.subscriber_id.isin(df_pred_ls)][df_nr.columns[1]] > 0)
    fp_nr = np.sum(df_nr[df_nr.subscriber_id.isin(df_pred_ls)][df_nr.columns[1]] == 0)
    tn_nr = np.sum(df_nr[df_nr.subscriber_id.isin(df_pred_ls) == False][df_nr.columns[1]] == 0)
    fn_nr = np.sum(df_nr[df_nr.subscriber_id.isin(df_pred_ls) == False][df_nr.columns[1]] > 0)

    prec_nr = tp_nr/(tp_nr+fp_nr)
    recall_nr = tp_nr/(tp_nr+fn_nr)
    F1_nr = 2 * prec * recall_pred/(prec + recall_pred)# 2*(precision * recall)/(precision + recall)

    #TODO all measure only not relevant subscribers

    print('true_prec: ', true_prec, '\npred_prec: ', pred_prec,
          '\ntrue_prec_a: ', true_prec_a, '\npred_prec_a: ', pred_prec_a,
          '\ntp, fp, tn, fn: ', tp, fp, tn, fn,
          '\nprec: ', prec,
          '\nF1: ', F1)
    acc = (tp +tn)/(tp+fp+tn+fn)

    return acc, pred_prec, true_prec, recall_pred, tp,fp,tn, fn, \
           pred_prec_a, true_prec_a, recall_pred_a, tp_a, fp_a, tn_a, fn_a, F1, \
           tp_nr, fp_nr, tn_nr, fn_nr, prec_nr, recall_nr, F1_nr, df_label_cp



# ===========================
#       Load data
# ===========================
domain_notifications_date = pd.read_csv(fn_domain_notifications_date)
domain_subscribers = pd.read_csv(fn_domain_subscribers, low_memory=False)
domain_subscribers_click = pd.read_csv(fn_domain_subscribers_click)

# ===========================
# Use only active domain subscribers and notifications
# ===========================

curr_domain_subscribers = domain_subscribers[domain_subscribers.domain_id==domain_id].copy()
curr_domain_notifications_date = domain_notifications_date[domain_notifications_date.domain_id==domain_id].copy()

curr_domain_subscribers['min_created_at'] = curr_domain_subscribers['created_at'].copy()
curr_domain_subscribers.loc[(curr_domain_subscribers['created_at']>curr_domain_subscribers['created_at.1'])==True, 'min_created_at'] = curr_domain_subscribers.loc[(curr_domain_subscribers['created_at']>curr_domain_subscribers['created_at.1'])==True, 'created_at.1'].copy()


# Reduce to only active subscribers
active_domain_subscribers_click = \
    domain_subscribers_click[domain_subscribers_click.subscriber_id.isin(domain_subscribers.subscriber_id)].copy()

# ===========================
#       Merge
# ===========================
# create key of subscriber x notifications
k1 = active_domain_subscribers_click[['subscriber_id']].drop_duplicates().copy()
if INCLUDE_USERS_WO_CLICK_HISTORY:
    k1 = curr_domain_subscribers[['subscriber_id']].drop_duplicates().copy()
k2 = curr_domain_notifications_date[['id']].copy() # sql already grouped and removed duplicates
k1['key'] = 1
k2['key'] = 1
k1k2 = pd.merge(k1, k2, on='key').drop("key", 1)
# Merge curr_domain_notifications_date into the key
k1k2_n = pd.merge(k1k2, curr_domain_notifications_date, on='id', how='outer', suffixes=('','_n'))

df = pd.pivot_table(curr_domain_subscribers[['subscriber_id','min_created_at']], index='subscriber_id', values = 'min_created_at', aggfunc=np.min)
df.reset_index(inplace=True)
active_subscriber_min_created = df[df.subscriber_id.isin(k1.subscriber_id)]
active_subscriber_min_created['min_created_at_hr'] = active_subscriber_min_created['min_created_at'].apply(lambda x: float(x[11:13]))
k1k2_n_s = pd.merge(k1k2_n, active_subscriber_min_created, on='subscriber_id', suffixes=('','_s'))
# Merge active_domain_subscribers_click into the key
k1k2_n_s_c = pd.merge(k1k2_n_s, active_domain_subscribers_click, left_on=['subscriber_id', 'id'], right_on=['subscriber_id', 'notification_id'], how='outer', suffixes=('','_c'))
k1k2_n_s_c.loc[:, 'is_open'] = 0
k1k2_n_s_c.loc[k1k2_n_s_c.cnt_clicks>0, 'is_open'] = 1
k1k2_n_s_c.loc[:, 'is_notify'] = 1
k1k2_n_s_c.loc[(k1k2_n_s_c.min_created_at>k1k2_n_s_c.send_at), 'is_notify'] = 0


print('### domain_id:', domain_id, '\n### domain_name: ', domain_notifications_date[domain_notifications_date.domain_id==domain_id].domain_name.unique()
      ,'\n### Active subscribers:' , len(active_domain_subscribers_click.subscriber_id.unique())
      ,'\n### Total subscribers:' , curr_domain_subscribers.shape[0])

if SPLIT_TRAIN_TEST_BY_DOY:
    # TODO for i in np.arange(SPLIT_TRAIN_TEST_BY_DOY, k1k2_n_s_c.ts_dayofyear.max()):
    #   df_test1, df_train1 = split_by_day_of_year(k1k2_n_s_c, i)
    # df_test2, df_train1, df_train1_label = split_by_day_of_year(k1k2_n_s_c, SPLIT_TRAIN_TEST_BY_DOY)
    # df_test21, df_test1, df_test1_label = split_by_day_of_year(df_test2, 116)

    # TODO change name df_test_2, df_test21, df_test_1 to df_aftr_split1, df_aftr_split2, df_between_split1_and2, respectively
    df_test2, df_train1 = split_by_day_of_year(k1k2_n_s_c, SPLIT_TRAIN_TEST_BY_DOY)
    df_test21, df_test1 = split_by_day_of_year(df_test2, 116)

#Q:What are the strongest days of the week?
#A:Friday and Saturday and Sunday. All the others are weekdays

if KEEP_WEEKDAY_ONLY:
    df_test = keep_week_day_only(df_test1)
    df_train = keep_week_day_only(df_train1)

    df_test2_noweekends = keep_week_day_only(df_test2) #todo change name to different
    df_test21_noweekends = keep_week_day_only(df_test21) #todo change name to different

# ====================== Apply Aggs ==========================

df_test_agg = agg_df_by_subscriber_dow_hr(df_test)
df_train_agg = agg_df_by_subscriber_dow_hr(df_train)

df_test_agg_hr = agg_df_by_subscriber_hr(df_test)
df_train_agg_hr = agg_df_by_subscriber_hr(df_train)


# ===========================
#       Modeling
# ===========================
if CNVRT_AGG_HR_TO_DIST:
    df_test_dist = cnvrt_is_open_hr_agg_to_dist(df_test_agg_hr)
    df_train_dist = cnvrt_is_open_hr_agg_to_dist(df_train_agg_hr)

# ===========================
#       Predict & Test
# ===========================
res = pd.DataFrame()
for doy_val in range(FIRST_TEST_DOY, LAST_TEST_DOY+1):
    if PREDICT_BY_DIST:

        # Get labels
        df_test2_labels = create_label_columns_doy_hr(df_test2)
        df_test2_noweekends_labels = create_label_columns_doy_hr(df_test2_noweekends)

        t = df_test2[df_test2.ts_dayofyear==doy_val]
        t_noweekend = df_test2_noweekends[df_test2_noweekends.ts_dayofyear == doy_val]

        df_test2_labels_hr = create_label_columns_hr(t)
        df_test2_noweekends_labels_hr = create_label_columns_hr(t_noweekend)

        #pred_ts_hour = 14
        for pred_ts_hour in t.ts_hour.unique():
            ls_subscriber_at_ts_hr, df_train_dist_pred, ls_relevant_subscriber = predict(df_train_dist, int(pred_ts_hour))

            if RUN_RAND:
                ls_subscriber_at_ts_hr, df_train_dist_pred, ls_relevant_subscriber = predict_rand(df_train_dist, int(pred_ts_hour))

            if PREDICT_BY_SUBSCRIPTION_HOUR:
                #df_test_dist_ = pd.merge(df_test_dist, active_subscriber_min_created, on='subscriber_id', suffixes=('', '_s'))
                df_train_dist_ = pd.merge(df_train_dist, active_subscriber_min_created, on='subscriber_id', suffixes=('', '_s'))
                ls_subscriber_at_ts_hr, df_train_dist_pred, ls_relevant_subscriber = predict_by_subscription_hour(df_train_dist_, int(pred_ts_hour))



            if TEST_PREDICT_DIST:

                #acc, pred_prec, true_prec, recall_pred, tp, fp, tn, fn, pred_prec_a, true_prec_a, recall_pred_a, tp_a, fp_a, tn_a, fn_a, F1, df_label_cp \
                acc, pred_prec, true_prec, recall_pred, tp, fp, tn, fn, \
                pred_prec_a, true_prec_a, recall_pred_a, tp_a, fp_a, tn_a, fn_a, F1, \
                tp_nr, fp_nr, tn_nr, fn_nr, prec_nr, recall_nr, F1_nr, df_label_cp \
                    = test_predict(ls_subscriber_at_ts_hr, df_test2_noweekends_labels_hr[['subscriber_id', 'label_is_open_'+str(int(pred_ts_hour))]], ls_relevant_subscriber)
                #x = [acc, pred_prec, true_prec, recall_pred, tp,fp,tn, fn, pred_prec_a, true_prec_a, recall_pred_a, tp_a, fp_a, tn_a, fn_a]
                res1 = pd.DataFrame(
                    {'doy_val':[doy_val], 'pred_ts_hour': [pred_ts_hour], 'acc': [acc], 'pred_prec': [pred_prec], 'true_prec': [true_prec], 'recall_pred': [recall_pred], 'tp': [tp],
                     'fp': [fp], 'tn': [tn], 'fn': [fn], 'pred_prec_a': [pred_prec_a], 'true_prec_a': [true_prec_a],
                     'recall_pred_a': [recall_pred_a], 'tp_a': [tp_a], 'fp_a': [fp_a], 'tn_a': [tn_a], 'fn_a': [fn_a], 'F1': [F1],
                     'tp_nr' : [tp_nr], 'fp_nr' : [fp_nr], 'tn_nr' : [tn_nr], 'fn_nr' : [fn_nr], 'prec_nr' : [prec_nr], 'recall_nr' : [recall_nr], 'F1_nr' : [F1_nr]})
                res = res.append(res1)
                df_label_cp.to_csv('results/lbl_pred_rnd'+ str(RUN_RAND) + str(PREDICT_BY_SUBSCRIPTION_HOUR) +'_domain_{}_ts_{}_doy_{}_hr_{}_F1_{}.csv'.format(domain_id, datetime.datetime.utcnow().date(), doy_val, str(int(pred_ts_hour)), str(F1)))
                df_train_dist_pred.to_csv('results/df_pred_rnd' + str(RUN_RAND) + str(PREDICT_BY_SUBSCRIPTION_HOUR) + '_domain_{}_ts_{}_doy_{}_hr_{}_F1_{}.csv' \
                        .format(domain_id, datetime.datetime.utcnow().date(), doy_val, str(int(pred_ts_hour)), str(F1)[:5]))
res.to_csv('tmp_res_rnd'+ str(RUN_RAND) + str(PREDICT_BY_SUBSCRIPTION_HOUR) +'_dm_{}_ts_{}.csv'.format(domain_id,datetime.datetime.utcnow().date()))
print('# clicks without subscriber_id is ', np.sum(active_domain_subscribers_click.subscriber_id.isin(curr_domain_subscribers[curr_domain_subscribers.channel_type_id!=2].subscriber_id.unique())))
print('### domain_id:', domain_id, '\n### domain_name: ', domain_notifications_date[domain_notifications_date.domain_id==domain_id].domain_name.unique()
      ,'\n### Active subscribers:' , len(active_domain_subscribers_click.subscriber_id.unique())
      ,'\n### Total subscribers:' , curr_domain_subscribers.shape[0]
      ,'\n### Relevant subscribers in test:' , len(ls_relevant_subscriber))


print('bbb')

