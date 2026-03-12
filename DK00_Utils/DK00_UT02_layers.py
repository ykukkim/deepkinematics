import torch
from math import sqrt
from torch import nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from DK00_Utils.DK00_UT00_config import CONSTANTS as C

"Utils"
class SPL(nn.Module):
    def __init__(self, input_size, number_joints, joint_size, num_layers, hidden_units, predict_arms = False,  predict_head = False, sparse=False):
        super(SPL, self).__init__()

        self.predict_arms = predict_arms
        self.predict_head = predict_head

        self.input_size = input_size
        self.joint_size = joint_size

        self.num_joints = number_joints
        if self.predict_head and self.predict_arms:
            self.skeleton = C.FK_skeleton_Full
            self.prediction_order_temp = list(C.FK_EVAL_JOINTS_FUll)
        elif self.predict_head and not self.predict_arms:
            self.skeleton = C.FK_skeleton_NoArms
            self.prediction_order_temp =list(C.FK_EVAL_JOINTS_NoArms)
        elif not self.predict_head and not self.predict_arms:
            self.skeleton = C.FK_skeleton_NoArmsHead
            self.prediction_order_temp = list(C.FK_EVAL_JOINTS_NoArmsHead)

        self.human_size = self.num_joints * self.joint_size
        self.sparse_spl = False

        self.per_joint_layers = num_layers
        self.per_joint_units = hidden_units

        self.kinematic_tree = self._build_kinematic_tree()
        self.prediction_order, self.indexed_skeleton = self._process_kinematic_tree()
        self.joint_predictors = nn.ModuleDict(self._build_joint_predictors())

    # maps joint IDs to their parent IDs and names
    def _build_kinematic_tree(self):
        kinematic_tree = dict()
        for joint_list in self.skeleton:
            for joint_entry in joint_list:
                parent_list_ = [joint_entry[0]] if joint_entry[0] > -1 else []
                kinematic_tree[joint_entry[1]] = [parent_list_, joint_entry[1], joint_entry[2]]
        return kinematic_tree

    # This function determines the order in which joints should be predicted,
    # ensuring parents are predicted before children.
    def _process_kinematic_tree(self):
        def get_all_parents(parent_list, parent_id, tree):
            if parent_id not in parent_list:
                parent_list.append(parent_id)
                for parent in tree[parent_id][0]:
                    get_all_parents(parent_list, parent, tree)

        indexed_skeleton = dict()
        prediction_order = self.prediction_order_temp
        for joint_id in prediction_order:
            joint_entry = self.kinematic_tree[joint_id]
            if self.sparse_spl:
                new_entry = joint_entry
            else:
                parent_list_ = list()
                if len(joint_entry[0]) > 0:
                    get_all_parents(parent_list_, joint_entry[0][0], self.kinematic_tree)
                new_entry = [parent_list_, joint_entry[1], joint_entry[2]]
            indexed_skeleton[joint_id] = new_entry

        return prediction_order, indexed_skeleton

    # For each joint, a neural network is created to predict its position
    # based on its context and its parent joints' predictions.
    def _build_joint_predictors(self):
        joint_predictors = {}
        for joint_key in self.prediction_order:
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_key]
            layers = []
            input_size = self.input_size + self.joint_size * len(parent_joint_ids)
            for _ in range(self.per_joint_layers):
                layers.append(nn.Linear(input_size, self.per_joint_units))
                layers.append(nn.ReLU())
                input_size = self.per_joint_units
            layers.append(nn.Linear(input_size, self.joint_size))
            joint_predictors[str(joint_id)] = nn.Sequential(*layers)
        return joint_predictors

    # In the forward pass, predictions are made hierarchically,
    # ensuring each joint's prediction is based on its parents' predictions.
    def forward(self, context):
        joint_predictions = {}

        for joint_key in self.prediction_order:
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_key]

            joint_inputs = [context]
            for parent_joint_id in parent_joint_ids:
                joint_inputs.append(joint_predictions[parent_joint_id])

            joint_input_tensor = torch.cat(joint_inputs, dim=-1)
            joint_predictions[joint_id] = self.joint_predictors[str(joint_id)](joint_input_tensor)

        pose_prediction = torch.cat(list(joint_predictions.values()), dim=-1)
        assert pose_prediction.shape[-1] == self.human_size, "Prediction not matching with the skeleton."
        return pose_prediction

class SkipConnection(nn.Module):
    """
    A simple skip connection for an MLP.
    """
    def __init__(self, input_dim, output_dim):
        super(SkipConnection, self).__init__()
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None

    def forward(self, x, residual):
        if self.projection is not None:
            residual = self.projection(residual)
        return x + residual

"Models"

class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size=None, bidirectional=False, dropout=0.0, learn_init_state=False, use_batch_norm=False):
        """
        An LSTM-RNN.
        :param input_size: Input size.
        :param hidden_units: List of hidden sizes for each layer.
        :param num_layers: How many layers to use.
        :param output_size: If given, a dense layer is automatically added to map to this size.
        :param bidirectional: If the RNN should be bidirectional.
        :param dropout: Dropout applied directly to the inputs (off by default).
        :param learn_init_state: Whether to learn the initial hidden state.
        """
        super(RNNLayer, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.input_size = input_size
        self.learn_init_state = learn_init_state
        self.input_drop = nn.Dropout(p=dropout) if dropout > 0.0 and self.num_layers > 1 else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(self.input_size) if use_batch_norm else nn.Identity()

        # Adjust hidden_units to match the number of layers if necessary
        if len(hidden_units) == 1:
            self.hidden_units = [hidden_units[0]] * self.num_layers
        else:
            self.hidden_units = hidden_units

        # Initialize hidden states if learn_init_state is True
        if learn_init_state:
            self.to_init_state_h = nn.ModuleList([
                nn.Linear(self.input_size, self.num_directions * self.hidden_units[i]) for i in range(self.num_layers)
            ])
            self.to_init_state_c = nn.ModuleList([
                nn.Linear(self.input_size, self.num_directions * self.hidden_units[i]) for i in range(self.num_layers)
            ])
            self.init_parameters()  # Initialize parameters

        # Define LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            lstm_input_size = input_size if i == 0 else self.hidden_units[i - 1] * self.num_directions
            self.lstm_layers.append(
                nn.LSTM(input_size=lstm_input_size,
                        hidden_size=self.hidden_units[i],
                        num_layers=1,  # Single layer because we are stacking manually
                        bidirectional=bidirectional,
                        batch_first=True,
                        dropout=dropout if i < self.num_layers - 1 else 0)
            )

        # Output layer
        if output_size is not None:
            self.to_out = nn.Linear(self.hidden_units[-1] * self.num_directions, output_size)
        else:
            self.to_out = nn.Identity()

    def init_parameters(self):
        # Initialize the parameters of the linear layers for initial states
        for layer in self.to_init_state_h + self.to_init_state_c:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def cell_init(self, inputs_):
        """Return the initial state of the cell for each LSTM layer."""
        batch_size = inputs_.shape[0]
        h0, c0 = [], []

        for i in range(self.num_layers):
            # Define the state size for the current layer
            state_size = (self.num_directions, batch_size, self.hidden_units[i])

            if self.learn_init_state:
                # Initialize hidden states using learned parameters
                h_init = self.to_init_state_h[i](inputs_[:, 0, :].view(batch_size, -1)).view(state_size)
                c_init = self.to_init_state_c[i](inputs_[:, 0, :].view(batch_size, -1)).view(state_size)
            else:
                # Default initialization with zeros
                h_init = torch.zeros(state_size, device=inputs_.device)
                c_init = torch.zeros(state_size, device=inputs_.device)

            h0.append(h_init)
            c0.append(c_init)

        return h0, c0

    def forward(self, x, seq_lengths):
        """
        Forward pass.
        :param x: A tensor of shape (N, F, input_size).
        :param seq_lengths: A tensor of shape (N,) indicating the true sequence length for each batch entry.
        :return: The output of the RNN.
        """
        batch_size, seq_length, _ = x.shape

        # Apply dropout and batch normalization
        x = self.input_drop(x)
        x = x.view(-1, x.size(-1))  # Reshape for batch normalization if necessary
        x = self.batch_norm(x)
        x = x.view(batch_size, seq_length, -1)  # Reshape back to original shape

        # Initialize hidden states
        h0, c0 = self.cell_init(x)

        # Pass through each LSTM layer
        lstm_out = x
        for i, lstm_layer in enumerate(self.lstm_layers):
            packed_input = pack_padded_sequence(lstm_out, seq_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
            hidden_states = (h0[i], c0[i])  # Use the appropriate initial states for this layer
            packed_output, _ = lstm_layer(packed_input, hidden_states)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_length)

        # Apply the output layer if present
        output = self.to_out(lstm_out)  # (N, F, output_size)

        return output

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.LSTM) and not mode:
                # Don't set the RNN into eval mode to avoid an exception thrown when using the RNN together with IEF.
                # This should not have an effect as train and eval mode in RNNs is exactly the same.
                continue
            module.train(mode)
        return self

class MLPLayer(nn.Module):
    """
    An MLP mapping from input size to output size going through multiple hidden dense layers.
    Each hidden layer can have a different number of units.
    Uses PReLU and can be configured to apply dropout and batch normalization.
    """
    def __init__(self, input_size, embedding_size, dropout=0.0, use_batch_norm=False,skip_connection=False):
        super(MLPLayer, self).__init__()

        if isinstance(embedding_size, list) is False:
            embedding_size = [embedding_size]

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        current_input_size = input_size

        # Create hidden layers with optional skip connections
        for i in range(len(embedding_size)):
            # Create linear layer
            self.layers.append(nn.Linear(current_input_size, embedding_size[i]))

            # Optionally add batch normalization
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(embedding_size[i]))

            # Add activation layer
            self.layers.append(nn.PReLU())

            # Optionally add dropout
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

            # Add projection layer if skip connections are enabled and not the first layer
            if skip_connection and i > 0:
                # Corrected: Use previous embedding size as input to projection
                self.projections.append(
                    nn.Linear(embedding_size[i-1], embedding_size[i]) if embedding_size[i-1] != embedding_size[i] else None
                )
            else:
                self.projections.append(None)

            # Update current input size for the next layer
            current_input_size = embedding_size[i]

        self.hidden_to_output = nn.Linear(current_input_size, embedding_size[-1])


    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        # Reshape input for processing
        x = x.view(-1, x.size(-1))

        residual = x  # Initialize residual for skip connections

        # Apply hidden layers with optional skip connections
        projection_index = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # Apply skip connection after a complete block of operations
                if self.skip_connection and projection_index > 0 and self.projections[projection_index - 1] is not None:
                    projected_residual = self.projections[projection_index - 1](residual)
                    x = projected_residual + x  # Add skip connection
                residual = x  # Update residual after projection
                projection_index += 1

            x = layer(x)

        # Reshape back to original shape
        x = x.view(batch_size, seq_length, -1)

        # Apply output layer
        x = self.hidden_to_output(x)

        return x

"TRANSFORMER"

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1, skip_connection=True, window_size=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.skip_connection = skip_connection
        self.window_size = window_size

    def forward(self, x, padding_mask=None):
        T = x.size(1)
        attn_mask = None

        if self.window_size:
            attn_mask = generate_local_attention_mask(seq_len=T, window_size=self.window_size).to(x.device)

        residual = x
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=padding_mask)
        x = residual + self.dropout(attn_out) if self.skip_connection else attn_out
        x = self.norm1(x)

        residual = x
        ff_out = self.feed_forward(x)
        x = residual + self.dropout(ff_out) if self.skip_connection else ff_out
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ff_hidden_dim, dropout=0.1, skip_connection=True, window_size=None):
        super().__init__()

        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dim=embed_dim[i],
                    num_heads=num_heads[i],
                    ff_hidden_dim=ff_hidden_dim[i],
                    dropout=dropout,
                    skip_connection=skip_connection,
                    window_size=window_size
                )
            )
            if i < num_layers - 1:
                self.projections.append(nn.Linear(embed_dim[i], embed_dim[i + 1]))

    def forward(self, x, padding_mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, padding_mask=padding_mask)
            if i < len(self.projections):
                x = self.projections[i](x)
        return x

def generate_local_attention_mask(seq_len, window_size):
    """
    Generates a (T, T) attention mask for local self-attention.
    Masked positions are set to True (masked out by MultiheadAttention).
    """
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False  # allow attention in the local window
    return mask  # shape: (T, T)

def get_key_padding_mask(x):
    """
    x: Tensor of shape (N, T, D)
    Returns a mask of shape (N, T), with True where input is padding
    """
    return (x.abs().sum(dim=-1) == 0)  # True for padded positions

"Graphic Convolution"
class GraphConvolution(nn.Module):
    """
    Adapted from: https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    Implements a basic graph convolution layer.
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.att = nn.Parameter(torch.FloatTensor(node_n, node_n))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support) # (batch_size, node_n, out_features)
        if self.bias is not None:
            output += self.bias
        return output

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'

class GC_Block(nn.Module):
    """
    Defines a residual block for the GCN.
    """

    def __init__(self, in_features, drop_out, bias=True, node_n=48):
        super(GC_Block, self).__init__()
        self.in_features = in_features

        self.gc1 = GraphConvolution(in_features = in_features, out_features=in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features = in_features, out_features=in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.dropout = nn.Dropout(drop_out)
        self.activation = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.gc2(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        y = self.dropout(y)

        return y + x

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.in_features})'

class GCN(nn.Module):
    """
    Graph Convolutional Network with residual blocks.
    """

    def __init__(self, input_feature, hidden_feature, drop_out, num_stage, node_n):
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_features = input_feature, out_features=hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = nn.ModuleList(
            [GC_Block(hidden_feature, drop_out=drop_out, node_n=node_n) for _ in range(num_stage)]
        )

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.dropout = nn.Dropout(drop_out)
        self.activation = nn.Tanh()

    def forward(self, x, is_out_resi=True):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        y = self.dropout(y)

        for block in self.gcbs:
            y = block(y)

        y = self.gc7(y)
        if is_out_resi:
            y += x
        return y