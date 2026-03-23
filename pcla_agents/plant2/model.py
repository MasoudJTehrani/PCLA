import logging

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
)

import timm

logger = logging.getLogger(__name__)

class HFLM(nn.Module):
    def __init__(self, config_net, config_all):
        super().__init__()
        self.config_all = config_all
        self.config_net = config_net

        # 0:padding, 1: vehicle, 2: pedestrian, 3: static, 4: stop_sign, 5: traffic_light, 6: emergency vehicle
        self.object_types = 7  # 6 objects and 1 padding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y
        self.fc_attributes = 4

        precisions = [
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_angle", 4),
            self.config_all.model.pre_training.get("precision_speed", 4),
        ]

        self.vocab_size = [2**i for i in precisions]
        self.vocab_size[0] = int((1 + self.config_all.model.training.get("range_factor_front", 1))/2*self.vocab_size[0])

        # model
        config = AutoConfig.from_pretrained(
            self.config_net.hf_checkpoint
        )  # load config from hugging face model
        self.n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)

        # TODO why do we still need the other embeddings?
        self.model.embeddings.word_embeddings = None
        self.model.pooler = None

        self.input_bev = self.config_all.model.training.get("input_bev", False)
        if self.input_bev:
            self.bev_encoder = timm.create_model("resnet18", pretrained=True, num_classes=512)

        # token embedding
        self.tok_emb = nn.ParameterList(
            nn.Linear(self.num_attributes, self.n_embd)
            for _ in range(self.object_types)
        )

        self.wp_rep = self.config_all.model.waypoints.representation
        self.wp_gen = self.config_all.model.waypoints.generator

        assert (self.wp_rep in ["waypoints", "path+2hot", "path+wps"]), f"Waypoint type {self.wp_rep} not supported"
        assert (self.wp_gen in ["singlegru", "multigru", "linear"]), f"Waypoint type {self.wp_gen} not supported"

        self.wp_len = self.config_all.model.waypoints.wps_len if self.wp_rep != "path+2hot" else 0
        self.path_len = self.config_all.model.waypoints.path_len if self.wp_rep != "waypoints" else 0

        if self.wp_gen == "singlegru":
            if self.wp_rep == "waypoints":
                num_tokens = 1
            else:
                num_tokens = 2

        elif self.wp_rep == "path+2hot":
            num_tokens = self.path_len + 1

        elif self.wp_rep == "path+wps":
            num_tokens = self.wp_len + self.path_len

        elif self.wp_rep == "waypoints":
            num_tokens = self.wp_len

        self.wp_token = nn.Parameter(torch.randn(num_tokens, self.n_embd))

        if self.config_net.get("use_dropout", False):
            self.drop = nn.Dropout(config_net.embd_pdrop)

        self.route_emb = nn.Linear(20*2, self.n_embd)

        self.speed_emb = nn.Embedding(4, self.n_embd)

        self.input_ego_speed = self.config_all.model.training.get("input_ego_speed", False)
        if self.input_ego_speed:
            self.ego_speed_emb = nn.Linear(1, self.n_embd)

        # # decoder head forecasting
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.n_embd, n_out)
                for n_out in self.vocab_size
            ]
        )

        # Waypoints
        if self.wp_rep != "path+2hot":
            if self.wp_gen == "linear":
                self.wp_generator = LinearWaypoints(self.n_embd)
            elif self.wp_gen == "multigru":
                self.wp_generator = GRUWaypointsPredictorInterFuser(self.n_embd, self.wp_len, self.config_all.model.waypoints.gru_hidden_size, 0)
            elif self.wp_gen == "singlegru":
                self.wp_generator = SingleGRUWaypoints(self.n_embd, self.wp_len)

        # Path
        if self.wp_rep != "waypoints":
            if self.wp_gen == "linear":
                self.path_generator = LinearWaypoints(self.n_embd)
            elif self.wp_gen == "multigru":
                self.path_generator = GRUWaypointsPredictorInterFuser(self.n_embd, self.path_len, self.config_all.model.waypoints.gru_hidden_size, 0)
            elif self.wp_gen == "singlegru":
                self.path_generator = SingleGRUWaypoints(self.n_embd, self.path_len)

        # Speed
        if self.wp_rep == "path+2hot":
            self.speed_classifier = nn.Linear(self.n_embd, self.config_all.model.waypoints.bins_speed)  # TODO im paper schauen

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.Embedding)
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
                elif pn.endswith("_ih") or pn.endswith("_hh") or (len(pn.split("_"))>= 2 and pn.split("_")[-2] in ["ih", "hh"]): # Added for gru, which has weight_ih_l0 etc
                    print("cfg optim",pn)
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
        batch_idxs = batch["idxs"]
        x_batch_objs = batch["x_objs"]
        route_batch = batch["route_original"]
        speed_limit_batch = batch["speed_limit"]

        # Remove type and embed
        embedding = torch.zeros((*x_batch_objs.shape[:-1], self.n_embd), device=x_batch_objs.device)
        for i in range(len(self.tok_emb)):
            embedding[x_batch_objs[..., 0] == i] = self.tok_emb[i](x_batch_objs[x_batch_objs[..., 0] == i, 1:]) # remove first element of row bcs its class idx

        embedding = embedding[batch_idxs] # Restore batch shape

        # Add axis in second dim (B x 1 x 512)
        route_tok = self.route_emb(route_batch.flatten(1))[:, None]
        embedding = torch.cat((route_tok, embedding), dim=1) # Add route to front

        # Add speed
        speed_tok = self.speed_emb(speed_limit_batch)[:, None]
        embedding = torch.cat((speed_tok, embedding), dim=1) # Add speed limit to front

        # How many tokens at the front before objects
        remove_idxs = 2

        # Add ego_speed:
        if self.input_ego_speed:
            ego_speed_tok = self.ego_speed_emb(batch["input_ego_speed"][:, None]) # Linear input needs to be (10, 1)
            ego_speed_tok = ego_speed_tok[:, None] # Add dim for cat
            embedding = torch.cat((ego_speed_tok, embedding), dim=1)
            remove_idxs += 1

        if self.input_bev:
            bev_tok = self.bev_encoder(batch["BEV"])[:, None]
            embedding = torch.cat((bev_tok, embedding), dim=1)
            remove_idxs += 1

        # Add wp tokens to front
        wp_tokens = self.wp_token.expand(embedding.shape[0], *self.wp_token.shape)
        embedding = torch.cat((wp_tokens, embedding), dim=1)
        remove_idxs += self.wp_token.shape[0]

        # embedding dropout
        if self.config_net.get("use_dropout", False):
            embedding = self.drop(embedding)

        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions

        # Object Forecasting
        if batch["y_objs"] is not None:
            targets = batch["y_objs"][batch_idxs]
            targets = [targets[..., i].flatten() for i in range(self.fc_attributes)] # Tensor to list of tensors

            logits = x[:, remove_idxs:]
            logits = [
                self.heads[i](logits).flatten(end_dim=-2)
                for i in range(self.fc_attributes)
            ]
        else:
            targets = None
            logits = None

        # Planning
        pred_path = None
        pred_wps = None
        pred_speed = None

        if self.wp_gen == "singlegru":
            if self.wp_rep != "path+2hot":
                pred_wps = self.wp_generator(x[:, 0, :])
            if self.wp_rep != "waypoints":
                pred_path = self.path_generator(x[:, 1, :])

        else:
            if self.wp_rep != "path+2hot":
                pred_wps = self.wp_generator(x[:, :self.wp_len, :])
            if self.wp_rep != "waypoints":
                pred_path = self.path_generator(x[:, self.wp_len:self.wp_len+self.path_len, :])

        if self.wp_rep == "path+2hot":
            if self.wp_gen == "singlegru":
                pred_speed = self.speed_classifier(x[:, 0, :])
            else:
                pred_speed = self.speed_classifier(x[:, self.path_len, :])

        pred_plan = (pred_path, pred_wps, pred_speed)

        return logits, targets, pred_plan, attn_map


class SingleGRUWaypoints(nn.Module):
    def __init__(self, n_embd,num_wps):
        super().__init__()
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=2, hidden_size=64)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(64, 2)

        self.num_wps = num_wps
    
    # TODO this is really weird
    def forward(self, token):
        z = self.wp_head(token)

        output_wp = []

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype)
        x = x.type_as(z)

        # autoregressive generation of output waypoints
        for _ in range(self.num_wps):
            x_in = x # TODO torch.cat([x, target_point], dim=1)
            z = self.wp_decoder(x_in, z)
            # TODO wp_relu ?!
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        return torch.stack(output_wp, dim=1)


class LinearWaypoints(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.wp_decoder = nn.Linear(n_embd, 2)
    
    def forward(self, tokens):
        diffs = self.wp_decoder(tokens)
        return torch.cumsum(diffs, 1)


class GRUWaypointsPredictorInterFuser(nn.Module):
  """
  A version of the waypoint GRU used in InterFuser.
  It embeds the target point and inputs it as hidden dimension instead of input.
  The scene state is described by waypoints x input_dim features which are added as input instead of initializing the
  hidden state.
  """

  def __init__(self, input_dim, waypoints, hidden_size, target_point_size):
    super().__init__()
    self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
    if target_point_size > 0:
      self.encoder = nn.Linear(target_point_size, hidden_size)
    self.target_point_size = target_point_size
    self.hidden_size = hidden_size
    self.decoder = nn.Linear(hidden_size, 2)
    self.waypoints = waypoints

  def forward(self, x, target_point=None):
    bs = x.shape[0]
    if self.target_point_size > 0:
      z = self.encoder(target_point).unsqueeze(0)
    else:
      z = torch.zeros((1, bs, self.hidden_size), device=x.device)
    output, _ = self.gru(x, z)
    output = output.reshape(bs * self.waypoints, -1)
    output = self.decoder(output).reshape(bs, self.waypoints, 2)
    output = torch.cumsum(output, 1)
    return output



if __name__=="__main__":
    import yaml
    # Read YAML file
    with open("PlanT/config/config.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    with open("PlanT/config/model/PlanT.yaml", 'r') as stream:
        plnt = yaml.safe_load(stream)

    plnt["training"]["learning_rate"] = float(plnt["training"]["learning_rate"])

    # plnt["waypoints"] = {"type": "add_pathwp"}

    cfg["model"] = plnt

    cfg["visualize"] = False

    cfg["trainset_size"] = 1

    # cfg["newinput"] = True

    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value

    cfg = DictAsMember(cfg)

    # from dataset_3_NPC_WPS import PlanTDataset
    from dataset import PlanTDataset
    # from dataset_3 import generate_batch
    ds = PlanTDataset("/home/simon/PlanT_2_cleanup/PlanT_2_2025_07_24/data", cfg)

    from dataset import generate_batch
    batch = generate_batch([ds[i] for i in range(10,20)])


    model = HFLM(cfg.model.network, cfg)#.cuda()

    model.configure_optimizers(cfg.model.training)
    # model.eval()

    res = model(batch)

    print(batch["x_objs"].tolist())

    print([x.shape if x is not None else "None" for x in res[2] ])

    loss = sum(x.sum() for x in res[0]) + sum(x.sum() for x in res[2] if x is not None)

    loss.backward()

    # for name, param in model.named_parameters():
    #     if name in ["model.embeddings.word_embeddings.weight", "model.pooler.dense.weight", "model.pooler.dense.bias"]:
    #         param.requires_grad = False

    for name, p in model.named_parameters():
        if p.grad is None:
            print(name)