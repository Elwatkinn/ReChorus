import torch
import torch.nn as nn
from utils.layers import MLP_Block
from models.BaseContextModel import ContextCTRModel
import torch.nn.functional as F


class final_mlpBase(object):
    def parse_model_args_finalmlp(parser):
        return parser

    def _define_init(self, args, corpus,
                 embedding_dim=10,
                 mlp1_hidden_units=[256,256,256],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0.0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[256,256,256],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0.0,
                 mlp2_batch_norm=False,
                 use_fs=True,
                 use_iamlp=True,
                 fs_hidden_units=[256],
                 fs1_context=['user_id'],
                 fs2_context=['item_id'],
                 num_heads=16,
                # embedding_dim=10,
                #  mlp1_hidden_units=[400],
                #  mlp1_hidden_activations="ReLU",
                #  mlp1_dropout=0.4,
                #  mlp1_batch_norm=True,
                #  mlp2_hidden_units=[800],
                #  mlp2_hidden_activations="ReLU",
                #  mlp2_dropout=0.2,
                #  mlp2_batch_norm=True,
                #  use_fs=True,
                #  fs_hidden_units=[800],
                #  fs1_context=['user_id'],
                #  fs2_context=['item_id'],
                #  num_heads=10,
                 **kwargs):
        self.embedding_dim = embedding_dim
        self.use_fs = use_fs
        self.use_iamlp = use_iamlp
        self.fs1_context = fs1_context
        self.fs2_context = fs2_context
        self.embedding_dict = nn.ModuleDict()
        for f in self.context_features:
            self.embedding_dict[f] = nn.Embedding(self.feature_max[f],self.embedding_dim) if f.endswith('_c') or f.endswith('_id') else\
                    nn.Linear(1,self.embedding_dim,bias=False)
        self.feature_dim = self.embedding_dim * len(self.embedding_dict)
        print('embdding: ',embedding_dim)
        print('feature_dim: ',self.feature_dim)
        print('context_features: ',self.context_features)
        # MLP
        self.mlp1 = MLP_Block(input_dim=self.feature_dim,
                        output_dim=None,
                        hidden_units=mlp1_hidden_units,
                        hidden_activations=mlp1_hidden_activations,
                        dropout_rates=mlp1_dropout,
                        batch_norm=mlp1_batch_norm)
        self.mlp2 = MLP_Block(input_dim=self.feature_dim,
                        output_dim=None,
                        hidden_units=mlp2_hidden_units,
                        hidden_activations=mlp2_hidden_activations,
                        dropout_rates=mlp2_dropout,
                        batch_norm=mlp2_batch_norm)
        # fs
        # print(self.embedding_dim)
        if self.use_fs:
            self.fs_module = FeatureSelection(self.feature_dim,
                                     self.embedding_dim, 
                                     fs_hidden_units,
                                     fs1_context,
                                     fs2_context,
                                     self.feature_max)
        if use_iamlp:
            self.fusion_module = InteractionAggregationMLP(mlp1_hidden_units[-1],
                                        mlp2_hidden_units[-1],output_dim=1)
        else:
            self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1],
                                        mlp2_hidden_units[-1],output_dim=1,num_heads=num_heads)
        self.apply(self.init_weights)
        
        
    def get_inputs(self,feed_dict):
        item_ids = feed_dict['item_id']
        batch_size, item_num = item_ids.shape
		# embedding
        context_vectors = [self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') 
						  else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
						  for f in self.context_features]
        context_vectors = torch.stack([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1) 
							for v in context_vectors], dim=-2) # batch size * item num * feature num * feature dim
        return context_vectors.flatten(start_dim=-2)
        
    def forward(self, feed_dict):
        # print(feed_dict)
        flat_emb=self.get_inputs(feed_dict)
        # print(flat_emb.shape)
        if self.use_fs:
            feat1, feat2 = self.fs_module(feed_dict, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        feat1=feat1.view(feat1.shape[0],feat1.shape[2])
        feat2=feat2.view(feat2.shape[0],feat2.shape[2])
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        out_dict={}
        out_dict['prediction'] = y_pred
        # print(111)
        return out_dict

class final_mlpCTR(ContextCTRModel, final_mlpBase):
    reader, runner = 'ContextReader', 'CTRRunner'
    extra_log_args = ['loss_n']
    
    @staticmethod
    def parse_model_args(parser):
        parser = final_mlpBase.parse_model_args_finalmlp(parser)
        return ContextCTRModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextCTRModel.__init__(self, args, corpus)
        self._define_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = final_mlpBase.forward(self, feed_dict)
        out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
        out_dict['label'] = feed_dict['label'].view(-1)
        return out_dict

class FeatureSelection(nn.Module):
    def __init__(self, feature_dim, embedding_dim, fs_hidden_units=[], 
                 fs1_context=[], fs2_context=[], feature_maxn=dict()):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        self.fs2_context = fs2_context
        self.embedding_dim = embedding_dim
        self.feature_maxn = feature_maxn

        # fs1 embedding
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_emb = nn.ModuleList([
                nn.Embedding(feature_maxn[ctx], embedding_dim) if (ctx.endswith("_c") or ctx.endswith('_id')) and ctx in feature_maxn
                else nn.Linear(1, embedding_dim) if ctx.endswith("_f")
                else ValueError(f"Undefined context {ctx}")
                for ctx in fs1_context
            ])

        # fs2 embedding
        if len(fs2_context) == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_emb = nn.ModuleList([
                nn.Embedding(feature_maxn[ctx], embedding_dim) if (ctx.endswith("_c") or ctx.endswith('_id')) and ctx in feature_maxn
                else nn.Linear(1, embedding_dim) if ctx.endswith("_f")
                else ValueError(f"Undefined context {ctx}")
                for ctx in fs2_context
            ])

        # Gate networks
        self.fs1_gate = MLP_Block(
            input_dim=embedding_dim * max(1, len(fs1_context)),
            output_dim=feature_dim,
            hidden_units=fs_hidden_units,
            hidden_activations="ReLU",
            output_activation="Sigmoid",
            batch_norm=False
        )
        self.fs2_gate = MLP_Block(
            input_dim=embedding_dim * max(1, len(fs2_context)),
            output_dim=feature_dim,
            hidden_units=fs_hidden_units,
            hidden_activations="ReLU",
            output_activation="Sigmoid",
            batch_norm=False
        )

    def _get_ctx_emb(self, context_list, ctx_emb_modules, feed_dict, flat_emb):
        emb_list = []
        for i, ctx in enumerate(context_list):
            if ctx.endswith('_c') or ctx.endswith('_id'):
                ctx_emb = ctx_emb_modules[i](feed_dict[ctx])
            else:
                ctx_emb = ctx_emb_modules[i](feed_dict[ctx].float().unsqueeze(-1))
            if len(ctx_emb.shape) == 2:
                # 扩展维度，与 flat_emb 对齐
                ctx_emb = ctx_emb.unsqueeze(1).repeat(1, flat_emb.size(1), 1)
            emb_list.append(ctx_emb)
        return torch.cat(emb_list, dim=-1)

    def forward(self, feed_dict, flat_emb):
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0),  flat_emb.size(1), 1)
        else:
            fs1_input = self._get_ctx_emb(self.fs1_context, self.fs1_ctx_emb, feed_dict, flat_emb)
        # gt1 = self.fs1_gate(fs1_input) * 2
        gt1 = F.softmax(self.fs1_gate(fs1_input), dim=-1)
        feature1 = flat_emb * gt1

        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0),  flat_emb.size(1), 1)
        else:
            fs2_input = self._get_ctx_emb(self.fs2_context, self.fs2_ctx_emb, feed_dict, flat_emb)
        # gt2 = self.fs2_gate(fs2_input) * 2
        gt2 = F.softmax(self.fs2_gate(fs2_input), dim=-1)
        feature2 = flat_emb * gt2

        return feature1, feature2



class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(x.shape[0], self.num_heads, self.head_x_dim)
        head_y = y.view(x.shape[0], self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), self.w_xy.view(self.num_heads, self.head_x_dim, -1))
                          .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output +=  xy.sum(dim=1)
        return output.squeeze(-1)
    
class InteractionAggregationMLP(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1):
        super(InteractionAggregationMLP, self).__init__()

        self.fusion_mlp = MLP_Block(
            input_dim = x_dim+y_dim,
            output_dim = output_dim,
            hidden_units=[512,256],
            hidden_activations="ReLU"
        )

    def forward(self, x, y):
        fusion_input = torch.cat([x, y], dim=-1)
        return self.fusion_mlp(fusion_input)
    