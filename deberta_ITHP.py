from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
from ITHP import ITHP
import global_configs
from global_configs import DEVICE

TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM
print(f'{__name__}: Text Dim = {TEXT_DIM}, Acoustic Dim = {ACOUSTIC_DIM}, Visual Dim = {VISUAL_DIM}')

class ITHP_DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.config = config
        self.pooler = BertPooler(config)
        model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.model = model.to(DEVICE)
        ITHP_args = {
            'X0_dim': TEXT_DIM,
            'X1_dim': ACOUSTIC_DIM,
            'X2_dim': VISUAL_DIM,
            'B0_dim': multimodal_config.B0_dim,
            'B1_dim': multimodal_config.B1_dim,
            'inter_dim': multimodal_config.inter_dim,
            'max_sen_len': multimodal_config.max_seq_length,
            'drop_prob': multimodal_config.drop_prob,
            'p_beta': multimodal_config.p_beta,
            'p_gamma': multimodal_config.p_gamma,
            'p_lambda': multimodal_config.p_lambda,
        }

        self.ITHP = ITHP(ITHP_args)

        self.expand = nn.Sequential(
            nn.Linear(multimodal_config.B1_dim, TEXT_DIM),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.init_weights()
        self.beta_shift = multimodal_config.beta_shift

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
    ):
        embedding_output = self.model(input_ids)
        x = embedding_output[0]
        b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.ITHP(x, visual, acoustic)
        h_m = self.expand(b1)
        acoustic_vis_embedding = self.beta_shift * h_m
        sequence_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + x)
        )
        pooled_output = self.pooler(sequence_output)

        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1


class ITHP_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dberta = ITHP_DebertaModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
    ):

        pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.dberta(
            input_ids,
            visual,
            acoustic,
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1
