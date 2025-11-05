import logging

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
)

logger = logging.getLogger(__name__)


class HFLM(nn.Module):
    def __init__(self, config_net, config_all):
        super().__init__()
        self.config_all = config_all
        self.config_net = config_net

        self.object_types = 8 # WP (CLS), Car, Route, Walker*, Static*, Padding  (* are new)
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y

        precisions = [
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_angle", 4),
            self.config_all.model.pre_training.get("precision_speed", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
        ]

        self.vocab_size = [2**i for i in precisions]

        # model
        config = AutoConfig.from_pretrained(
            self.config_net.hf_checkpoint
        )  # load config from hugging face model
        n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)

        # Remove word embeddings
        self.model.embeddings.word_embeddings = None
        self.model.pooler = None

        # token embedding
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        self.cls_embed = nn.Embedding(self.object_types, n_embd)
        self.drop = nn.Dropout(config_net.embd_pdrop)

        # decoder head forecasting
        # one head for each attribute type -> we have different precision per attribute
        self.heads = nn.ModuleList(
            [
                nn.Linear(n_embd, self.vocab_size[i])
                for i in range(self.num_attributes)
            ]
        )

        # wp (CLS) decoding
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=64) # Change: 64 instead of 65 because we removed flags
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(64, 2)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = torch.nn.Linear
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("_ih") or pn.endswith("_hh"):
                    # all recurrent weights will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("_emb") or "_token" in pn:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer


    def forward(self, batch):

        target_point = batch["target_point"]

        x_objs = batch["x_objs"] # This is a flat tensor of our input objects
        batch_idxs = batch["idxs"] # This restores the batch shape of the objects

        # Linear projection
        x_embs = self.tok_emb(x_objs[..., 1:]) # Remove first attribute (class)

        # Class embeddings
        x_embs[x_objs[..., 0] == 0] = 0 # Clear tokens with class 0, these are the CLS token for waypoints, we only want the embedding
        x_embs[x_objs[..., 0] == 5] = 0 # Same for padding
        for i in torch.arange(self.object_types, dtype=torch.long, device=x_embs.device):
            x_embs[x_objs[..., 0] == i] += self.cls_embed(i)

        # Restore Batch x N_objs x n_embd shape
        embedding = x_embs[batch_idxs]

        # embedding dropout
        embedding = self.drop(embedding) # Change: In the original code this was skipped due to a mistake in the code

        # Transformer Encoder; use embedding for hugging face model and get output states
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions

        # Forecasting
        if batch["y_objs"] is not None:
            
            # Restore batch structure 
            targets = batch["y_objs"][batch["idxs"]]

            # Filter x using target class 10, then calculate logits for each attribute
            logits = [self.heads[i](x[targets[..., 0] != 10]) for i in range(self.num_attributes)]

            # Apply same filter to targets and drop class attribute
            targets = targets[targets[..., 0] != 10][..., 1:]

        else:
            logits = None
            targets = None

        # Waypoints
        z = self.wp_head(x[:, 0, :]) # First token per sample in x is always the CLS object

        output_wp = []

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype)
        x = x.type_as(z)

        # autoregressive generation of output waypoints
        for _ in range(self.config_all.model.training.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.wp_decoder(x_in, z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return logits, targets, pred_wp, attn_map


# Debugging
if __name__=="__main__":
    import yaml
    # Read YAML file
    with open("/home/simon/PlanTUpdate/config/config.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    with open("/home/simon/PlanTUpdate/config/model/PlanT.yaml", 'r') as stream:
        plnt = yaml.safe_load(stream)

    cfg["model"] = plnt

    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value

    cfg = DictAsMember(cfg)

    from dataset import PlanTDataset, generate_batch

    ds = PlanTDataset("/home/simon/PDM-Lite-DS/Town12/PedestrianCrossing", cfg)

    batch = generate_batch([ds[416], ds[214]])


    model = HFLM(cfg.model.network, cfg)#.cuda()

    res = model(batch)

    print(res)

    train_cfg = {"weight_decay": 1e-3,
                 "learning_rate": 1e-3,
                 "betas": (0.9, 0.999)}
    
    # Test that all params are matched etc
    model.configure_optimizers(DictAsMember(train_cfg))

    # Used to check that there are no unused params
    fake_loss = sum(x.sum() for x in res[0]) + res[2].sum()

    fake_loss.backward()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("No grad:", name)
