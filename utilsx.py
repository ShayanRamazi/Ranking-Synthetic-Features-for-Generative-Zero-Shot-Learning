# import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
from sklearn.cluster import KMeans

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def init_seeds(seed=0):
    np.random.seed(seed)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[torch.nonzero(self.train_mapped_label == i),:].numpy(), axis=0)

        n_cluster = opt.n_clusters
        real_proto = torch.zeros(n_cluster * self.train_cls_num, self.feature_dim)
        for i in range(self.train_cls_num):
            sample_idx = (self.train_mapped_label == i).nonzero().squeeze()
            if sample_idx.numel() == 0:
                real_proto[n_cluster * i: n_cluster * (i+1)] = torch.zeros(n_cluster, self.feature_dim)
            else:
                real_sample_cls = self.train_feature[sample_idx, :]
                y_pred = KMeans(n_clusters=n_cluster, random_state=3).fit_predict(real_sample_cls)
                for j in range(n_cluster):
                    real_proto[n_cluster*i+j] = torch.from_numpy(real_sample_cls[torch.nonzero(torch.from_numpy(y_pred)==j),:].mean(dim=0).cpu().numpy())
        self.real_proto = real_proto
    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()]
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()]
        val_unseen_loc = fid['val_unseen_loc'][()]
        test_seen_loc = fid['test_seen_loc'][()]
        test_unseen_loc = fid['test_unseen_loc'][()]
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc]
            self.train_label = label[trainval_loc]
            self.test_unseen_feature = feature[test_unseen_loc]
            self.test_unseen_label = label[test_unseen_loc]
            self.test_seen_feature = feature[test_seen_loc]
            self.test_seen_label = label[test_seen_loc]
        else:
            self.train_feature = feature[train_loc]
            self.train_label = label[train_loc]
            self.test_unseen_feature = feature[val_unseen_loc]
            self.test_unseen_label = label[val_unseen_loc]

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:

            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
        train_feature_val = torch.from_numpy(feature[trainval_loc]).float()
        train_label_val = torch.from_numpy(label[trainval_loc]).long()
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        print(self.train_class)
        train_feature_all = torch.cat((train_feature_val, self.train_feature), 0)
        train_label_all = torch.cat((train_label_val, self.train_label), 0)
        allseenclasses = torch.from_numpy(np.unique(train_label_all.numpy()))
        print("ssss")
        init_seeds(520)
        #print(np.random.get_state()[1][0])
        #520
        #AWA way :20 shots:64 16
        #flo way :41 shots:16 10
        #CUB way :75 shots
        ways = 20 * 2
        shots = 64
        is_first = True
        select_feats = []
        select_atts = []
        select_labels = []
        select_labels = torch.LongTensor(ways * shots)
        selected_classes = np.random.choice(list(allseenclasses), ways, True)
        for i in range(len(selected_classes)):
            idx = (train_label_all == selected_classes[i]).nonzero()[0]
            idx = train_label_all.eq(selected_classes[i]).nonzero().squeeze()
            select_instances = np.random.choice(idx, shots, False)
            lam = np.random.beta(5, 1)
           
            if i < ways / 2:
                for j in range(shots):
                    feat = train_feature_all[select_instances[j]]
                    att = self.attribute[train_label_all[select_instances[j]]]
                    feat = feat.unsqueeze(0)
                    att = att.unsqueeze(0)
                    if is_first:
                        is_first = False
                        select_feats = feat
                        select_atts = att
                    else:
                        select_feats = torch.cat((select_feats, feat), 0)
                        select_atts = torch.cat((select_atts, att), 0)
                    select_labels[i * shots + j] = i
            else:
                for j in range(shots):
                    feat = train_feature_all[select_instances[j]]
                    att = self.attribute[train_label_all[select_instances[j]]]
                    feat = feat.unsqueeze(0)
                    feat = lam * feat + (1 - lam) * select_feats[int(i - ways / 2) * shots + j]
                    att = att.unsqueeze(0)
                    att = lam * att + (1 - lam) * select_atts[int(i - ways / 2) * shots + j]
                    
                    select_feats = torch.cat((select_feats, feat), 0)
                    select_atts = torch.cat((select_atts, att), 0)
                    select_labels[i * shots + j] = i - int(ways / 2)
       
        noval_index = int(ways / 2) * shots
        dictx = {}
        ind=0
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
        for i in select_atts[noval_index:]:
            
            ind=ind+1
            output = 0
            for j in self.attribute[self.unseenclasses]:
                output = output + cos(i,j)#((i - j) ** 2).sum(axis=0)
            dictx[ind] = output
        x = dict(sorted(dictx.items(), key=lambda item: item[1]))
        #awa 1024
        #zsl flo 160 gzsl 
        best_items = list(x.items())[-1024:]
        la=True
        a=[]
        f=[]
        l = torch.LongTensor(1024)
        n=0
        
        for i in range(len(best_items)):
          if la:
            f.append(select_feats[best_items[i][0]])
            a.append(select_atts[best_items[i][0]])
            l[i]=n
            la=False
          else:
            f.append(select_feats[best_items[i][0]])
            a.append(select_atts[best_items[i][0]])
            if(best_items[i][1]!=best_items[i-1][1]):
                n=n+1
            l[i]=n
        
        f=torch.stack(f)
        a=torch.stack(a)
        self.attribute = torch.cat((self.attribute, torch.unique(a, dim=0)), 0)
        print(1)
        #awa 50
        #flo 102
        #cub 200
        self.train_label = torch.cat((self.train_label, l + 50), 0)
        self.train_feature = torch.cat((self.train_feature, f), 0)
        self.ntrain = self.train_feature.size()[0]
        allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)


    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]


    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att


    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att
