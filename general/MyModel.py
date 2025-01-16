import sys
sys.path.append("src\\")

from statistics import mean
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp
from utils import utils
from models.BaseModel import GeneralModel
'''
python main.py --model_name MyModel --reg 1e-6 --ssl 0.2 --temp 0.1 --dataset Grocery_and_Gourmet_Food
'''

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class MyModel(GeneralModel):
    reader = 'BaseReader'
    runner = 'MyRunner'
    extra_log_args = ['emb_size', 'hyper_num', 'leaky', 'gnn_layer',
                       'keepRate', 'reg', 'temp', 'ssl_reg']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--hyper_num', type=int, default=128,
                            help='number of hyperedges')
        parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
        parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
        parser.add_argument('--keepRate', default=0.5, type=float, help='keep rate of edges')
        parser.add_argument('--reg', default=1e-6, type=float, help='regularization')
        parser.add_argument('--temp', default=0.1, type=float, help='temperature')
        parser.add_argument('--ssl_reg', default=1e-3, type=float, help='regularization of ssl')

        return GeneralModel.parse_model_args(parser)

    @staticmethod
    def normalizeAdj(mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
    
    @staticmethod
    def build_adjmat(data, user, item):
        mat=sp.coo_matrix((np.ones_like(data['user_id']), (data['user_id'], data['item_id'])))
        a = sp.csr_matrix((user, user))
        b = sp.csr_matrix((item, item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        mat=mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hyper_num = args.hyper_num
        self.gnn_layer = args.gnn_layer
        self.leaky = args.leaky
        self.keepRate = args.keepRate
        self.reg = args.reg
        self.temp = args.temp
        self.ssl_reg = args.ssl_reg

        self.adj = self.build_adjmat(corpus.data_df['train'], self.user_num, self.item_num)

        self._define_params()
        #self.apply(self.init_weights)

    def _define_params(self):
        self.uEmbeds = nn.Parameter(init(torch.empty(self.user_num, self.emb_size)))
        self.iEmbeds = nn.Parameter(init(torch.empty(self.item_num, self.emb_size)))
        self.gcnLayer = GCNLayer(self.leaky)
        self.hgnnLayer = HGNNLayer(self.leaky)
        self.uHyper = nn.Parameter(init(torch.empty(self.emb_size, self.hyper_num)))
        self.iHyper = nn.Parameter(init(torch.empty(self.emb_size, self.hyper_num)))
        self.edgeDropper = SpAdjDropEdge()

    def forward(self, feed_dict):
        embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]
        gnnLats = []
        hyperLats = []
        uuHyper = self.uEmbeds @ self.uHyper
        iiHyper = self.iEmbeds @ self.iHyper

        for i in range(self.gnn_layer):
            temEmbeds = self.gcnLayer(self.edgeDropper(self.adj, self.keepRate), lats[-1])
            hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-self.keepRate), lats[-1][:self.user_num])
            hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-self.keepRate), lats[-1][self.user_num:])
            gnnLats.append(temEmbeds)
            hyperLats.append(torch.concat([hyperULat, hyperILat], dim=0))
            lats.append(temEmbeds + hyperLats[-1])

        embeds = sum(lats)
        uEmbeds, iEmbeds = embeds[:self.user_num], embeds[self.user_num:]
        ancEmbeds = uEmbeds[feed_dict['user_id'].long()]
        posEmbeds = iEmbeds[feed_dict['item_id'][:, 0].long()]
        negEmbeds = iEmbeds[feed_dict['item_id'][:, 1:].long()]
        
        return {'ancEmbeds': ancEmbeds, 'posEmbeds': posEmbeds, 'negEmbeds': negEmbeds, 'gnnLats': gnnLats, 'hyperLats': hyperLats}

    def loss(self, out_dict, feed_dict):
        ancEmbeds = out_dict['ancEmbeds']
        posEmbeds = out_dict['posEmbeds']
        negEmbeds = out_dict['negEmbeds']
        gcnEmbedsLst = out_dict['gnnLats']
        hyperEmbedsLst = out_dict['hyperLats']
        pos_rating=(ancEmbeds*posEmbeds).sum(dim=-1)
        neg_rating=(ancEmbeds*negEmbeds).sum(dim=-1)

        scoreDiff = utils.pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().mean()
        numerator = torch.exp(pos_rating / self.temp)
        denominator = numerator + torch.sum(torch.exp(neg_rating / self.temp), dim = 1)
        soft_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        sslLoss = 0
        for i in range(self.gnn_layer):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += utils.contrastLoss(embeds1[:self.user_num], embeds2[:self.user_num], torch.unique(feed_dict['user_id']), self.temp)+ utils.contrastLoss(embeds1[self.user_num:], embeds2[self.user_num:], torch.unique(feed_dict['item_id']), self.temp)
        
        sslLoss *= self.ssl_reg
        regLoss = utils.calcRegLoss(self) * self.reg
        alpha = 0.8
        loss = alpha * bprLoss + regLoss + sslLoss + (1-alpha) * soft_loss
        #loss = bprLoss + regLoss + sslLoss
        
        return loss
    
    def predict(self,batch):
        pre_keeprate=1.0
        embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]
        gnnLats = []
        hyperLats = []
        uuHyper = self.uEmbeds @ self.uHyper
        iiHyper = self.iEmbeds @ self.iHyper

        for i in range(self.gnn_layer):
            temEmbeds = self.gcnLayer(self.edgeDropper(self.adj, pre_keeprate), lats[-1])
            hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-pre_keeprate), lats[-1][:self.user_num])
            hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-pre_keeprate), lats[-1][self.user_num:])
            gnnLats.append(temEmbeds)
            hyperLats.append(torch.concat([hyperULat, hyperILat], dim=0))
            lats.append(temEmbeds + hyperLats[-1])

        embeds = sum(lats)
        uEmbeds, iEmbeds = embeds[:self.user_num], embeds[self.user_num:]
        m1=uEmbeds[batch['user_id'].long()]
        m2=iEmbeds[batch['item_id'].long()]
        allPreds=(m1[:, None, :] * m2).sum(dim=-1)

        return allPreds


class GCNLayer(nn.Module):
	def __init__(self, leaky):
		super(GCNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=leaky)

	def forward(self, adj, embeds):
		return (torch.spmm(adj, embeds))

class HGNNLayer(nn.Module):
	def __init__(self, leaky):
		super(HGNNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=leaky)
	
	def forward(self, adj, embeds):
		# lat = self.act(adj.T @ embeds)
		# ret = self.act(adj @ lat)
		lat = (adj.T @ embeds)
		ret = (adj @ lat)
		return ret

class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
