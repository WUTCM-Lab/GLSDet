import torch

class TransformerDecorator(torch.nn.Module):
    def __init__(self, pool_layer=None, add_bt=3, dim=10, eval_global=0):
        super(TransformerDecorator, self).__init__()
        self.encoder_layers = torch.nn.TransformerEncoderLayer(dim, 4, dim, 0.5)
        self.eval_global = eval_global
        self.add_bt = add_bt
        self.pool_layer = pool_layer
        self.training = True

    def forward_feats(self, feature):
        feature = self.pool_layer(feature.mean([-2, -1]))
        if self.training and self.add_bt > 0:
            pre_feature = feature
            feature = feature.unsqueeze(1)
            feature = self.encoder_layers(feature)
            # feature = self.encoder_layers(feature, feature, feature)
            feature = feature.squeeze(1)
            if self.add_bt:
                return torch.cat([pre_feature, feature], dim=0)
        return feature

    def forward(self, feature):
        feature = self.forward_feats(feature)
        return feature

if __name__ == '__main__':
    module = TransformerDecorator(dim=40)
    a = torch.rand(2, 8, 40)
    print(module(a).size)


