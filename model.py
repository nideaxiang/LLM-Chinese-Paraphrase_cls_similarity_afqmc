from torch import nn
from transformers import BertPreTrainedModel,BertModel
from transformers import AutoConfig



class BertForPairwiseCLS2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config,add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()
    def forward(self,x):
        bert_outputs=self.bert(**x)
        cls_output=bert_outputs.last_hidden_state[:,0,:]
        cls_output=self.dropout(cls_output)
        logits=self.classifier(cls_output)
        return logits



