import os
import sys
from collections import Counter, defaultdict
from itertools import chain

import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_fscore_support, \
    roc_curve
from sklearn.svm import SVC, LinearSVC

from helper import flatten_data
from load_sated import process_texts, process_vocabs, load_texts, load_users, load_sated_data_by_user, \
    SATED_TRAIN_USER, SATED_TRAIN_FR, SATED_TRAIN_ENG
from sated_nmt import build_nmt_model, words_to_indices, MODEL_PATH, OUTPUT_PATH
from sated_nmt_ranks import get_target_ranks, get_shadow_ranks, ranks_to_feats


# HELPER METHODS
def histogram_feats(ranks, bins=100, top_words=5000, num_words=5000, relative=False):
    if top_words < num_words:
        if bins == top_words:
            bins += 1
        top_words += 1

    range = (-num_words, top_words) if relative else (0, top_words)
    feats, _ = np.histogram(ranks, bins=bins, normed=False, range=range)
    return feats


def avg_rank_feats(ranks):
    avg_ranks = []
    for r in ranks:
        avg = np.mean(np.concatenate(r))
        avg_ranks.append(avg)
    return avg_ranks


def sample_with_ratio(a, b, heldout_ratio=0.5):
    if heldout_ratio == 0.:
        return a
    if heldout_ratio == 1.:
        return b

    if not isinstance(a, list):
        a = a.tolist()
        b = b.tolist()

    l1 = len(a)
    l2 = len(b)
    ratio = float(l2) / (l1 + l2)
    if heldout_ratio > ratio:
        # remove from a
        n = int(l2 / heldout_ratio)
        rest_l1 = n - l2
        return a[:rest_l1] + b
    elif heldout_ratio < ratio:
        # remove from b
        n = int(l1 / (1 - heldout_ratio))
        rest_l2 = n - l1
        return a + b[:rest_l2]
    else:
        return a + b


def get_indices_by_labels(sent_labels):
    sent_label_sum = [-np.sum(labels) for labels in sent_labels]
    return np.argsort(sent_label_sum)


def load_ranks_by_label(save_dir, num_users=5000, cross_domain=False, label=1):
    ranks = []
    labels = []
    y = []
    for i in range(num_users):
        save_path = save_dir + 'rank_u{}_y{}{}.npz'.format(i, label, '_cd' if cross_domain else '')
        if os.path.exists(save_path):
            f = np.load(save_path, allow_pickle=True)
            train_rs, train_ls = f['arr_0'], f['arr_1']
            ranks.append(train_rs)
            labels.append(train_ls)
            y.append(label)

    return ranks, labels, y


def load_all_ranks(save_dir, num_users=5000, cross_domain=False):
    ranks = []
    labels = []
    y = []

    train_label = 1
    train_ranks, train_labels, train_y = load_ranks_by_label(save_dir, num_users, cross_domain, train_label)
    ranks = ranks + train_ranks
    labels = labels + train_labels
    y = y + train_y

    test_label = 0
    test_ranks, test_labels, test_y = load_ranks_by_label(save_dir, num_users, cross_domain, test_label)
    ranks = ranks + test_ranks
    labels = labels + test_labels
    y = y + test_y

    return ranks, labels, np.asarray(y)


def ranks_to_feats(ranks, labels=None, prop=1.0, dim=100, num_words=5000, top_words=5000, shuffle=False,
                   rare=False, relative=False, user_data_ratio=0., heldout_ratio=0., num_users=300):
    if relative or rare:
        assert labels is not None
    X = []

    for i, user_ranks in enumerate(ranks):
        indices = np.arange(len(user_ranks))
        if relative or rare:
            user_labels = labels[i]
            assert len(user_labels) == len(user_ranks)
        else:
            user_labels = None

        r = []

        if 0. < user_data_ratio < 1. and i < num_users:
            l = len(user_ranks)
            for idx in range(l):
                user_ranks[idx] = np.clip(user_ranks[idx], 0, top_words)
                if relative:
                    assert len(user_ranks[idx]) == len(user_labels[idx])
                    user_ranks[idx] = user_ranks[idx] - user_labels[idx]

            train_l = int(l * user_data_ratio)
            train_ranks = user_ranks[:train_l]
            heldout_ranks = user_ranks[train_l:]
            for rank in sample_with_ratio(train_ranks, heldout_ranks, heldout_ratio):
                r.append(rank)
        else:
            if shuffle:
                np.random.seed(None)
                np.random.shuffle(indices)

            if rare:
                indices = get_indices_by_labels(user_labels)

            n = int(len(indices) * prop) + 1 if isinstance(prop, float) else prop
            for idx in indices[:n]:
                user_ranks[idx] = np.clip(user_ranks[idx], 0, top_words)
                if relative:
                    assert len(user_ranks[idx]) == len(user_labels[idx])
                    r.append(user_ranks[idx] - user_labels[idx])
                else:
                    r.append(user_ranks[idx])

        # print i, r
        if isinstance(r[0], int):
            print(i)
        else:
            r = np.concatenate(r)

        feats = histogram_feats(r, bins=dim, num_words=num_words, top_words=top_words, relative=relative)
        X.append(feats)

    return np.vstack(X)


# Attack 1: Average Rank Thresholding
def run_attack1(num_users=5000, dim=100, prop=1.0, user_data_ratio=0., attacker_knowledge=0.5,
                heldout_ratio=0., num_words=5000, top_words=5000, relative=False, rare=False,
                norm=True, scale=True, cross_domain=False, rerun=False):
    result_path = OUTPUT_PATH

    if dim > top_words:
        dim = top_words

    attack1_results_save_path = result_path + 'mi_data_dim{}_prop{}_{}{}_attack1.npz'.format(
        dim, prop, num_users, '_cd' if cross_domain else '')

    if not rerun and os.path.exists(attack1_results_save_path):
        f = np.load(attack1_results_save_path)
        X_train, y_train, X_test, y_test = [f['arr_{}'.format(i)] for i in range(4)]
    else:
        save_dir = result_path + 'target_{}{}/'.format(num_users, '_dr' if 0. < user_data_ratio < 1. else '')
        train_ranks, train_labels, train_y = load_ranks_by_label(save_dir, num_users, label=1)
        test_ranks, test_labels, test_y = load_ranks_by_label(save_dir, num_users, label=0)

        # Split into in/out to train classifier
        minimum = min(len(train_ranks), len(test_ranks))
        knowledge_prop = int(attacker_knowledge * minimum)
        in_train_ranks, in_train_labels, in_train_y = train_ranks[:knowledge_prop], train_labels[:knowledge_prop], \
                                                      train_y[:knowledge_prop]
        out_train_ranks, out_train_labels, out_train_y = test_ranks[:knowledge_prop], test_labels[:knowledge_prop], \
                                                         test_y[:knowledge_prop]
        in_test_ranks, in_test_labels, in_test_y = train_ranks[knowledge_prop:], train_labels[knowledge_prop:], \
                                                   train_y[knowledge_prop:]
        out_test_ranks, out_test_labels, out_test_y = test_ranks[knowledge_prop:], test_labels[knowledge_prop:], \
                                                      test_y[knowledge_prop:]

        # Convert to average rank features
        in_train_feat = avg_rank_feats(in_train_ranks)
        in_test_feat = avg_rank_feats(in_test_ranks)
        out_train_feat = avg_rank_feats(out_train_ranks)
        out_test_feat = avg_rank_feats(out_test_ranks)

        # Create dataset
        X_train, y_train = np.concatenate([in_train_feat, out_train_feat]), np.concatenate([in_train_y, out_train_y])
        X_test, y_test = np.concatenate([in_test_feat, out_test_feat]), np.concatenate([in_test_y, out_test_y])
        np.savez(attack1_results_save_path, X_train, y_train, X_test, y_test)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    clf = SVC()
    clf.fit(X_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(X_test.reshape(-1, 1))
    print("Attack 1 Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))

    ra_score = roc_auc_score(y_test, y_pred)
    print("Attack 1 ROC_AUC Score : %.2f%%" % (100.0 * ra_score))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    plt.figure(1)
    plt.plot(fpr, tpr, label='Attack 1')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('sateduser_attack1_roc_curve.png')


# Attack 2: Shadow Models on Rank Histograms
def run_attack2(num_exp=5, num_users=5000, dim=100, prop=1.0, user_data_ratio=0.,
                heldout_ratio=0., num_words=5000, top_words=5000, relative=False, rare=False, norm=True,
                scale=True, cross_domain=False, rerun=False):

    result_path = OUTPUT_PATH

    if dim > top_words:
        dim = top_words

    audit_save_path = result_path + 'mi_data_dim{}_prop{}_{}{}.npz'.format(
        dim, prop, num_users, '_cd' if cross_domain else '')

    if not rerun and os.path.exists(audit_save_path):
        f = np.load(audit_save_path, allow_pickle=True)
        X_train, y_train, X_test, y_test = [f['arr_{}'.format(i)] for i in range(4)]
    else:
        save_dir = result_path + 'target_{}{}/'.format(num_users, '_dr' if 0. < user_data_ratio < 1. else '')
        ranks, labels, y_test = load_all_ranks(save_dir, num_users)
        X_test = ranks_to_feats(ranks, prop=prop, dim=dim, top_words=top_words, user_data_ratio=user_data_ratio,
                                num_words=num_words, labels=labels, rare=rare, relative=relative,
                                heldout_ratio=heldout_ratio)

        X_train, y_train = [], []
        for exp_id in range(num_exp):
            save_dir = result_path + 'shadow_exp{}_{}/'.format(exp_id, num_users)
            ranks, labels, y = load_all_ranks(save_dir, num_users, cross_domain=cross_domain)
            feats = ranks_to_feats(ranks, prop=prop, dim=dim, top_words=top_words, relative=relative,
                                   num_words=num_words, labels=labels)
            X_train.append(feats)
            y_train.append(y)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        np.savez(audit_save_path, X_train, y_train, X_test, y_test)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    if norm:
        normalizer = Normalizer(norm='l2')
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)

    print(classification_report(y_pred=y_pred, y_true=y_test))

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    pres, recs, _, _ = precision_recall_fscore_support(y_test, y_pred)
    pre = pres[1]
    rec = recs[1]

    print('precision={}, recall={}, acc={}, auc={}'.format(pre, rec, acc, auc))

    ra_score = roc_auc_score(y_test, y_pred)
    print("Attack 1 ROC_AUC Score : %.2f%%" % (100.0 * ra_score))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    plt.figure(2)
    plt.plot(fpr, tpr, label='Attack 2')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('sateduser_attack2_roc_curve.png')

    return acc, auc, pre, rec


# Attack 3: Shadow Models for Sequence Classification
def run_attack3():
    return


if __name__ == '__main__':
    num_shadow_models = 4
    num_users = 300
    cross_domain_flag = False
    attacker_knowledge_ratio = 0.1

    # attacker knowledge = 50%
    run_attack1(num_users=num_users,
                attacker_knowledge=attacker_knowledge_ratio,
                rerun=True)

    print("....................................................................................................")

    # 4 shadow models, 300 users
    acc, auc, pre, rec = run_attack2(num_exp=num_shadow_models,
                                     num_users=num_users,
                                     cross_domain=cross_domain_flag,
                                     rerun=True)

