import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import TensorProduct
from e3nn.nn import Gate, BatchNorm
from e3nn.o3 import Linear, FullyConnectedTensorProduct
from torch_scatter import scatter
import math


class CosineCutoff(nn.Module):
    """
    Cosine cutoff function for smooth edge weighting
    """
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
        
    def forward(self, distances):
        """
        Args:
            distances: [E] - edge lengths
        Returns:
            [E] - cutoff weights in [0, 1]
        """
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).to(cutoffs.dtype)
        return cutoffs


class ExpNormalSmearing(nn.Module):
    """
    Exponential normal smearing with cosine cutoff
    More expressive than Gaussian smearing with trainable parameters
    """
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        """Initialize means and betas in exponential space"""
        start_value = torch.exp(torch.tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def forward(self, distances):
        """
        Args:
            distances: [E] - edge lengths
        Returns:
            [E, num_rbf] - radial basis functions
        """
        distances = distances.unsqueeze(-1)
        
        # Exponential transformation
        exp_dist = torch.exp(-self.alpha * distances)
        
        # RBF with trainable means and betas
        rbf = torch.exp(-self.betas * (exp_dist - self.means) ** 2)
        
        # Apply smooth cutoff
        cutoff_weights = self.cutoff_fn(distances.squeeze(-1)).unsqueeze(-1)
        
        return rbf * cutoff_weights
    
class GaussianSmearing(nn.Module):
    """
    Gaussian smearing for edge lengths (legacy, kept for compatibility)
    """
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        
        self.width = (stop - start) / (num_gaussians - 1)
    
    def forward(self, distances):
        """
        Args:
            distances: [E] - edge lengths
        Returns:
            [E, num_gaussians]
        """
        distances = distances.unsqueeze(-1)
        return torch.exp(-0.5 * ((distances - self.offset) / self.width) ** 2)
    
class SymmetricTensorProduct(nn.Module):
    """
    Efficient symmetric tensor product using e3nn's TensorProduct with
    symmetric instructions. This is more efficient than FullyConnectedTensorProduct
    for MACE-style many-body features.
    
    Implements: A ⊗ A ⊗ ... ⊗ A (n times) with symmetrization
    """
    def __init__(self, irreps_in, irreps_out, correlation_order, shared_weights=True):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.correlation_order = correlation_order
        
        if correlation_order == 1:
            # Just a linear mapping for 2-body
            self.tp = Linear(self.irreps_in, self.irreps_out)
        elif correlation_order == 2:
            # 3-body: A ⊗ A with symmetric coupling
            self.tp = self._build_symmetric_tp_order2()
        else:
            # Higher orders: iterative symmetric coupling
            self.tps = nn.ModuleList()
            current_irreps = self.irreps_in
            
            for i in range(correlation_order - 1):
                if i == correlation_order - 2:
                    # Last one outputs to irreps_out
                    tp = self._build_pairwise_symmetric_tp(
                        current_irreps, self.irreps_in, self.irreps_out
                    )
                else:
                    # Intermediate tensor products
                    tp = self._build_pairwise_symmetric_tp(
                        current_irreps, self.irreps_in, self.irreps_in
                    )
                    current_irreps = self.irreps_in
                
                self.tps.append(tp)
    
    def _build_symmetric_tp_order2(self):
        """Build symmetric tensor product for A ⊗ A"""
        # Create instructions for symmetric coupling only
        instructions = []
        
        for i, (mul_i, ir_i) in enumerate(self.irreps_in):
            for j, (mul_j, ir_j) in enumerate(self.irreps_in):
                # Only include i <= j for symmetry (no duplicate pairs)
                if i > j:
                    continue
                    
                for ir_out in ir_i * ir_j:
                    # Check if this irrep is in our output
                    for k, (mul_k, ir_k) in enumerate(self.irreps_out):
                        if ir_out == ir_k:
                            # Add instruction: (input1_idx, input2_idx, output_idx, mode, train, path_weight)
                            # mode: 'uvw' for standard, 'uvu' for symmetric (u=input1, v=input2, w=output)
                            mode = 'uvu' if i == j else 'uvw'
                            instructions.append((i, j, k, mode, True, 1.0))
        
        # Build tensor product with symmetric instructions
        return TensorProduct(
            self.irreps_in,
            self.irreps_in, 
            self.irreps_out,
            instructions=instructions,
            shared_weights=True,
            internal_weights=True
        )
    
    def _build_pairwise_symmetric_tp(self, irreps1, irreps2, irreps_out):
        """Build tensor product between two potentially different irreps"""
        instructions = []
        
        irreps1 = o3.Irreps(irreps1)
        irreps2 = o3.Irreps(irreps2)
        irreps_out = o3.Irreps(irreps_out)
        
        for i, (mul_i, ir_i) in enumerate(irreps1):
            for j, (mul_j, ir_j) in enumerate(irreps2):
                for ir_out in ir_i * ir_j:
                    for k, (mul_k, ir_k) in enumerate(irreps_out):
                        if ir_out == ir_k:
                            instructions.append((i, j, k, 'uvw', True, 1.0))
        
        return TensorProduct(
            irreps1,
            irreps2,
            irreps_out,
            instructions=instructions,
            shared_weights=True,
            internal_weights=True
        )
    
    def forward(self, x):
        """
        Args:
            x: [N, irreps_in.dim]
        Returns:
            [N, irreps_out.dim]
        """
        if self.correlation_order == 1:
            return self.tp(x)
        elif self.correlation_order == 2:
            return self.tp(x, x)
        else:
            # Iteratively apply tensor products
            result = x
            for tp in self.tps:
                result = tp(result, x)
            return result


class MACEConvolutionLayer(nn.Module):
    """
    MACE-inspired many-body message passing layer using efficient symmetric
    tensor products for higher-order interactions.
    
    This implementation uses o3.TensorProduct with symmetric instructions
    instead of FullyConnectedTensorProduct for better efficiency.
    
    Key equations from MACE paper:
    - Equation (8): A-features construction with spherical harmonics
    - Equation (10): B-features via symmetric tensor products
    - Equation (11): Message as linear combination of B-features
    
    References:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing 
        Neural Networks for Fast and Accurate Force Fields" NeurIPS 2022
    """
    def __init__(
        self, 
        irreps_node_input,
        irreps_node_hidden,
        irreps_sh,
        max_correlation_order=3,  # ν in paper (1=2-body, 2=3-body, 3=4-body)
        num_channels=128,
        edge_features_dim=16,
        num_radial_basis=8,
        hidden_dim=64,
        normalization='layer',
        use_self_connection=True
    ):
        super().__init__()
        
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        
        self.max_correlation_order = max_correlation_order
        self.num_channels = num_channels
        self.use_self_connection = use_self_connection
        
        # ============ Radial Network (R in Equation 8) ============
        radial_input_dim = num_radial_basis + edge_features_dim
        
        self.radial_mlp = nn.Sequential(
            nn.Linear(radial_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ============ Node Feature Embedding (W in Equation 8) ============
        # Project input features to scalar channels
        self.node_embedding = Linear(
            self.irreps_node_input,
            o3.Irreps(f"{num_channels}x0e")
        )
        
        # ============ A-features: Tensor Product with Spherical Harmonics ============
        # Build efficient tensor product for node_scalars ⊗ spherical_harmonics
        # This creates directional 2-body features
        self.a_features_tp = TensorProduct(
            o3.Irreps(f"{num_channels}x0e"),
            self.irreps_sh,
            self.irreps_node_hidden,
            instructions=[
                (i, j, k, 'uvw', True, 1.0)
                for i, (mul_i, ir_i) in enumerate([o3.Irrep('0e')] * num_channels)
                for j, (mul_j, ir_j) in enumerate(self.irreps_sh)
                for k, (mul_k, ir_k) in enumerate(self.irreps_node_hidden)
                if ir_i * ir_j == ir_k
            ],
            shared_weights=False,
            internal_weights=False
        )
        
        # Weights for A-features tensor product (controlled by radial network)
        self.a_tp_weights = nn.Linear(hidden_dim, self.a_features_tp.weight_numel)
        
        # ============ B-features: Symmetric Tensor Products ============
        # Create symmetric tensor products for different correlation orders
        # This is the key innovation: efficient many-body without explicit summation
        
        self.symmetric_contractions = nn.ModuleList()
        self.channel_mixing = nn.ModuleList()
        
        # Extract scalar and low-L irreps for tensor products
        # (higher L are expensive, MACE typically uses up to L=2)
        self.tp_irreps = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node_hidden 
            if ir.l <= 2
        ])
        
        for order in range(1, max_correlation_order + 1):
            # Symmetric tensor product for this correlation order
            symmetric_tp = SymmetricTensorProduct(
                irreps_in=self.tp_irreps,
                irreps_out=self.irreps_node_hidden,
                correlation_order=order,
                shared_weights=True
            )
            self.symmetric_contractions.append(symmetric_tp)
            
            # Channel mixing (linear projection)
            channel_mix = Linear(self.irreps_node_hidden, self.irreps_node_hidden)
            self.channel_mixing.append(channel_mix)
        
        # ============ Message Combination (Equation 11) ============
        # Combine B-features from different correlation orders
        self.message_combination = nn.ModuleList([
            Linear(self.irreps_node_hidden, self.irreps_node_hidden)
            for _ in range(max_correlation_order)
        ])
        
        # ============ Self-interaction and Update ============
        if use_self_connection:
            self.self_interaction = Linear(
                self.irreps_node_input,
                self.irreps_node_hidden
            )
        
        # Optional normalization
        if normalization == 'layer':
            from e3nn.nn import BatchNorm
            self.norm = BatchNorm(self.irreps_node_hidden)
        elif normalization == 'instance':
            from e3nn.nn import BatchNorm
            self.norm = BatchNorm(self.irreps_node_hidden, instance=True)
        else:
            self.norm = None
    
    def forward(
        self, 
        node_features,  # h^(t): [N, irreps_node_input.dim]
        edge_index,     # [2, E]
        edge_sh,        # Y_l^m: [E, irreps_sh.dim]
        edge_radial_embedding,  # [E, num_radial_basis]
        edge_attr=None, # [E, edge_features_dim]
        batch_mask=None # [N]
    ):
        """
        Forward pass implementing MACE's efficient many-body message passing.
        
        Returns:
            [N, irreps_node_hidden.dim] - updated node features
        """
        src, dst = edge_index
        num_nodes = node_features.shape[0]
        
        # ============ Step 1: Construct A-features (Equation 8) ============
        # A_i = Σ_j R(r_ij) Y(r̂_ij) ⊗ W h_j
        
        # Embed node features to scalars
        h_scalars = self.node_embedding(node_features)  # [N, num_channels]
        
        # Radial weighting
        if edge_attr is not None:
            radial_input = torch.cat([edge_radial_embedding, edge_attr], dim=-1)
        else:
            radial_input = edge_radial_embedding
        
        radial_features = self.radial_mlp(radial_input)  # [E, hidden_dim]
        
        # Get weights for tensor product from radial network
        tp_weights = self.a_tp_weights(radial_features)  # [E, weight_numel]
        
        # Edge-wise tensor product: h_j ⊗ Y(r̂_ij)
        edge_messages = self.a_features_tp(
            h_scalars[src],  # [E, num_channels]
            edge_sh,         # [E, irreps_sh.dim]
            tp_weights       # [E, weight_numel]
        )  # [E, irreps_node_hidden.dim]
        
        # Aggregate messages to nodes (Equation 8 summation)
        a_features = scatter(
            edge_messages,
            dst,
            dim=0,
            dim_size=num_nodes,
            reduce='mean'  # or 'sum' depending on normalization preference
        )  # [N, irreps_node_hidden.dim]
        
        # ============ Step 2: Construct B-features (Equation 10) ============
        # B^ν = Σ C^{LM}_{η_ν,lm} Π_{ξ=1}^ν (Σ_k w A)
        # Symmetric tensor products for different correlation orders
        
        # Extract features for tensor product (only up to L=2)
        a_for_tp = a_features  # Already in correct irreps from a_features_tp output
        
        b_features_list = []
        for order_idx, (symmetric_tp, channel_mix) in enumerate(
            zip(self.symmetric_contractions, self.channel_mixing)
        ):
            # Apply symmetric tensor product
            b_feat = symmetric_tp(a_for_tp)  # [N, irreps_node_hidden.dim]
            
            # Channel mixing
            b_feat = channel_mix(b_feat)
            
            b_features_list.append(b_feat)
        
        # ============ Step 3: Construct Messages (Equation 11) ============
        # m = Σ_ν W_{z_i} B^ν
        
        messages = None
        for b_feat, linear in zip(b_features_list, self.message_combination):
            msg_contribution = linear(b_feat)
            messages = msg_contribution if messages is None else messages + msg_contribution
        
        # ============ Step 4: Update (Equation 12) ============
        # h^{t+1} = U(h^t, m^t) with residual connection
        
        if self.use_self_connection:
            output = messages + self.self_interaction(node_features)
        else:
            output = messages
        
        # Normalization for stability
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class MACEBlock(nn.Module):
    """
    Complete MACE block that can be used as a drop-in replacement for 
    standard equivariant layers.
    
    Features:
    - Efficient many-body interactions via symmetric tensor products
    - Reduced layer count (2 layers achieve same expressiveness as 5+ EGNN layers)
    - Small receptive field (no need for deep stacking)
    """
    def __init__(
        self,
        irreps_node_input,
        irreps_node_hidden, 
        irreps_sh,
        max_correlation_order=3,  # 4-body interactions by default
        num_channels=128,
        edge_features_dim=16,
        num_radial_basis=8,
        hidden_dim=64,
        normalization='layer',
        use_self_connection=True
    ):
        super().__init__()
        
        self.max_correlation_order = max_correlation_order
        
        self.conv = MACEConvolutionLayer(
            irreps_node_input=irreps_node_input,
            irreps_node_hidden=irreps_node_hidden,
            irreps_sh=irreps_sh,
            max_correlation_order=max_correlation_order,
            num_channels=num_channels,
            edge_features_dim=edge_features_dim,
            num_radial_basis=num_radial_basis,
            hidden_dim=hidden_dim,
            normalization=normalization,
            use_self_connection=use_self_connection
        )
    
    def forward(self, h, edge_index, edge_sh, edge_radial_embedding, 
                edge_attr=None, batch_mask=None):
        """
        Args:
            h: [N, irreps_node_input.dim] - node features
            edge_index: [2, E] - edge connectivity
            edge_sh: [E, irreps_sh.dim] - spherical harmonics
            edge_radial_embedding: [E, num_radial_basis] - radial basis functions
            edge_attr: [E, edge_features_dim] - optional edge attributes
            batch_mask: [N] - batch indices for normalization
        
        Returns:
            [N, irreps_node_hidden.dim] - updated node features
        """
        return self.conv(
            h, edge_index, edge_sh, edge_radial_embedding, 
            edge_attr, batch_mask
        )
    
class SeparableSphericalConvolution(nn.Module):

    def __init__(self, irreps_in, irreps_out, irreps_sh, edge_features=0, 
                 hidden_dim=64, residual=True, normalization='layer', weight_mode='dynamic'):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.residual = residual and (self.irreps_in == self.irreps_out)
        self.weight_mode = weight_mode

        instructions = []
        for i, (mul_in, ir_in) in enumerate(self.irreps_in):
            for j, (mul_sh, ir_sh) in enumerate(self.irreps_sh):
                for k, (mul_out, ir_out) in enumerate(self.irreps_out):
                    if ir_out in ir_in * ir_sh:
                        instructions.append((i, j, k, 'uvw', True))
        
        if self.weight_mode =='dynamic':
        
            self.tp = TensorProduct(
                self.irreps_in,
                self.irreps_sh,
                self.irreps_out,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False
            )
            
            # MLP nhỏ gọn
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_features, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.tp.weight_numel)
            )
        elif self.weight_mode == 'scalar':
            self.tp = TensorProduct(
                self.irreps_in,
                self.irreps_sh,
                self.irreps_out,
                instructions=instructions,
                shared_weights=True,
                internal_weights=True
            )

            self.edge_mlp = nn.Sequential(
                    nn.Linear(edge_features, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(edge_features, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1) # Output a single scalar
                )
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}. "
                             f"Choose from 'dynamic', 'scalar'")     
           
        self.self_interaction = Linear(self.irreps_in, self.irreps_out)
        
        if normalization == 'layer':
            self.norm = BatchNorm(self.irreps_out)
        elif normalization == 'instance':
            self.norm = BatchNorm(self.irreps_out, instance=True)
        else:
            self.norm = None

    def forward(self, h, edge_index, edge_sh, edge_features, batch_mask=None):
        src, dst = edge_index

        if self.weight_mode == 'dynamic':
            edge_weights = self.edge_mlp(edge_features)
            messages = self.tp(h[src], edge_sh, edge_weights)
        
        elif self.weight_mode == 'scalar':
            base_messages = self.tp(h[src], edge_sh)
            edge_scalar_weights = self.edge_mlp(edge_features) # Shape: [E, 1]
            messages = base_messages * edge_scalar_weights # Broadcasting scales the messages
        else:
            raise ValueError(f"Invalid weight_mode '{self.weight_mode}' during forward pass.")
        
        aggregated = scatter(messages, dst, dim=0, dim_size=h.shape[0], reduce='mean')
        out = aggregated + self.self_interaction(h)
        
        if self.norm is not None:
            out = self.norm(out)
        if self.residual:
            out = out + h
            
        return out

class SphericalHarmonicsConvolution(nn.Module):
    """
    E(3) equivariant convolution layer using spherical harmonics
    with normalization for stability
    """
    def __init__(self, irreps_in, irreps_out, irreps_sh, edge_features=0, 
                 hidden_dim=64, residual=True, normalization='layer', weight_mode='dynamic'):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.residual = residual and (self.irreps_in == self.irreps_out)
        self.weight_mode = weight_mode
        
        # Tensor product: node features x spherical harmonics -> messages        
        if self.weight_mode =='dynamic':
            self.tp = FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_sh,
                self.irreps_out,
                shared_weights=False,
                internal_weights=False
            )
            
            # MLP nhỏ gọn
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_features, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.tp.weight_numel)
            )
        elif self.weight_mode == 'scalar':
            self.tp = FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_sh,
                self.irreps_out,
                shared_weights=True,
                internal_weights=True
            )

            self.edge_mlp = nn.Sequential(
                    nn.Linear(edge_features, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1) # Output a single scalar
                )
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}. "
                            f"Choose from 'dynamic', 'scalar'")     
        # Self-interaction
        self.self_interaction = Linear(self.irreps_in, self.irreps_out)
        
        # Normalization for stability
        if normalization == 'layer':
            self.norm = BatchNorm(self.irreps_out)
        elif normalization == 'instance':
            self.norm = BatchNorm(self.irreps_out, instance=True)
        else:
            self.norm = None

    def edge_model(self, source, edge_sh, edge_features):

        if self.weight_mode == 'dynamic':
            edge_weights = self.edge_mlp(edge_features)
            messages = self.tp(source, edge_sh, edge_weights)
        
        elif self.weight_mode == 'scalar':
            base_messages = self.tp(source, edge_sh)
            edge_scalar_weights = self.edge_mlp(edge_features) # Shape: [E, 1]
            messages = base_messages * edge_scalar_weights # Broadcasting scales the messages
        
        return messages
    
    def node_model(self,h, edge_index, messages):
        
        src, dst = edge_index
        
        aggregated = scatter(messages, dst, dim=0, dim_size=h.shape[0], reduce='mean')
        
        # Self-interaction
        out = aggregated + self.self_interaction(h)
        
        # Normalization for stability
        if self.norm is not None:
            out = self.norm(out)
        
        # Residual connection
        if self.residual:
            out = out + h
            
        return out
            
    def forward(self, h, edge_index, edge_sh, edge_features, 
                batch_mask=None):
        """
        Args:
            h: [N, irreps_in.dim] - node features
            edge_index: [2, E] - edge indices
            edge_sh: [E, irreps_sh.dim] - spherical harmonics
            edge_attr: [E, edge_attr_dim] - edge attributes
            edge_length_embedded: [E, edge_embed_dim] - embedded edge lengths
            batch_mask: [N] - batch indices for normalization
        """
        src, dst = edge_index
        
        messages = self.edge_model(h[src], edge_sh, edge_features)
        
        # Node model: aggregate and update
        out = self.node_model(h, edge_index, messages)
         
        return out
    
class SphericalHarmonicsBlock(nn.Module):
    """
    Equivariant block with spherical harmonics convolution
    """
    def __init__(self, irreps_hidden, irreps_sh, edge_features, hidden_dim=64,
                 residual=True, normalization='layer', conv_type='separable', weight_mode='dynamic'):
        super().__init__()
        
        if conv_type == 'separable':
            self.conv = SeparableSphericalConvolution(
                irreps_in=irreps_hidden,
                irreps_out=irreps_hidden,
                irreps_sh=irreps_sh,
                edge_features=edge_features,
                hidden_dim=hidden_dim,
                residual=residual,
                normalization=normalization,
                weight_mode=weight_mode
            )
            
        elif conv_type == 'full':
            self.conv = SphericalHarmonicsConvolution(
                irreps_in=irreps_hidden,
                irreps_out=irreps_hidden,
                irreps_sh=irreps_sh,
                edge_features=edge_features,
                hidden_dim=hidden_dim,
                residual=residual,
                normalization=normalization,
                weight_mode=weight_mode

            )
        else:
            raise ValueError(f"Unknown convolution type: {conv_type}. Choose 'separable' or 'full'.")
        
    def forward(self, h, edge_index, edge_sh, edge_features, 
                batch_mask=None):
        """
        Args:
            h: [N, irreps_hidden.dim]
            edge_index: [2, E]
            edge_sh: [E, irreps_sh.dim]
            edge_attr: [E, edge_attr_dim]
            edge_length_embedded: [E, edge_embed_dim]
            batch_mask: [N]
        """
        return self.conv(h, edge_index, edge_sh, edge_features, batch_mask)
    
class VectorOutputHead(nn.Module):
    """
    Output head for vector predictions using L=1 and L=2
    """
    def __init__(self, irreps_in, hidden_dim=64):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        
        # Extract L=1 and L=2 components
        self.irreps_l1 = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_in if l == 1])
        self.irreps_l2 = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_in if l == 2])
        
        if len(self.irreps_l1) > 0:
            self.l1_projection = Linear(self.irreps_l1, o3.Irreps("1x1o"))
        else:
            self.l1_projection = None
        
        if len(self.irreps_l2) > 0:
            self.l2_to_l1 = FullyConnectedTensorProduct(
                self.irreps_l2,
                o3.Irreps("1x1o"),
                o3.Irreps("1x1o"),
                shared_weights=True
            )
            self.coupling_vector = nn.Parameter(torch.randn(3) * 0.01)
        else:
            self.l2_to_l1 = None
        
        self.final_linear = nn.Linear(3, 3, bias=False)
        nn.init.xavier_uniform_(self.final_linear.weight, gain=0.001)
    
    def forward(self, features):
        """
        Args:
            features: [N, irreps_in.dim]
        Returns:
            [N, 3]
        """
        idx = 0
        l1_features = []
        l2_features = []
        
        for mul, (l, p) in self.irreps_in:
            dim = mul * (2 * l + 1)
            if l == 1:
                l1_features.append(features[:, idx:idx+dim])
            elif l == 2:
                l2_features.append(features[:, idx:idx+dim])
            idx += dim
        
        outputs = []
        
        if l1_features and self.l1_projection is not None:
            l1_feat = torch.cat(l1_features, dim=1)
            vec_from_l1 = self.l1_projection(l1_feat)
            outputs.append(vec_from_l1)
        
        if l2_features and self.l2_to_l1 is not None:
            l2_feat = torch.cat(l2_features, dim=1)
            coupling = self.coupling_vector.unsqueeze(0).expand(features.size(0), -1)
            vec_from_l2 = self.l2_to_l1(l2_feat, coupling)
            outputs.append(vec_from_l2)
        
        if len(outputs) == 0:
            return torch.zeros(features.size(0), 3, device=features.device)
        
        output = sum(outputs) / len(outputs)
        output = self.final_linear(output)
        
        return output


class ScalarOutputHead(nn.Module):
    """
    Output head for scalar predictions using L=0 and L=2
    """
    def __init__(self, irreps_in, output_dim, hidden_dim=64):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        
        # Extract L=0 and L=2 features
        self.irreps_l0 = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_in if l == 0])
        self.irreps_l2 = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_in if l == 2])
        
        if len(self.irreps_l2) > 0:
            self.l2_to_scalar = nn.Sequential(
                nn.Linear(self.irreps_l2.dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.l2_to_scalar = None
        
        combined_dim = self.irreps_l0.dim
        if self.l2_to_scalar is not None:
            combined_dim += hidden_dim
            
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, features):
        """
        Args:
            features: [N, irreps_in.dim]
        Returns:
            [N, output_dim]
        """
        idx = 0
        l0_features = []
        l2_features = []
        
        for mul, (l, p) in self.irreps_in:
            dim = mul * (2 * l + 1)
            if l == 0:
                l0_features.append(features[:, idx:idx+dim])
            elif l == 2:
                l2_features.append(features[:, idx:idx+dim])
            idx += dim
        
        l0_feat = torch.cat(l0_features, dim=1) if l0_features else torch.zeros(features.size(0), 0, device=features.device)
        l2_feat = torch.cat(l2_features, dim=1) if l2_features else None
        
        if l2_feat is not None and self.l2_to_scalar is not None:
            l2_scalar = self.l2_to_scalar(l2_feat)
            combined = torch.cat([l0_feat, l2_scalar], dim=1)
        else:
            combined = l0_feat
        
        return self.output_mlp(combined)

class EGNN_Spherical(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device ='cpu',
                 act_fn=nn.SiLU(), n_layers = 3, attention=False, 
                 out_node_nf=None, lmax=2, num_rbf=16,
                 max_radius=10.0, normalization='layer',
                 aggregation_method='mean', reflection_equiv=True,
                 rbf="expnormal", trainable_rbf=True, convolution_type='separable', weight_mode='dynamic'):
        super().__init__()

        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.lmax = lmax
        self.max_radius = max_radius
        self.num_rbf = num_rbf
        self.normalization = normalization
        self.convolution_type = convolution_type
        self.weight_mode = weight_mode

        irreps_hidden = o3.Irreps(
            f"{hidden_nf//2}x0e + {hidden_nf//4}x1o + {hidden_nf//8}x2e"
        )
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        # Edge embedding
        self.edge_embed_dim = num_rbf + in_edge_nf

        if rbf == 'expnormal':
            self.radial_basis = ExpNormalSmearing(
                cutoff=max_radius, num_rbf=num_rbf, trainable=trainable_rbf
            )
        else:
            self.radial_basis = GaussianSmearing(0.0, max_radius, num_rbf)

        #Input Embedding

        self.embedding = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, irreps_hidden[0].dim)
        )

        # Project to full irreps
        self.scalar_to_irreps = Linear(
            o3.Irreps(f"{irreps_hidden[0].dim}x0e"),
            irreps_hidden
        )

        # Convolution layer
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            self.conv_layers.append(
                SphericalHarmonicsBlock(
                irreps_hidden=irreps_hidden,
                irreps_sh=self.irreps_sh,
                edge_features=self.edge_embed_dim,
                hidden_dim = hidden_nf,
                residual=True,
                normalization=normalization,
                conv_type=self.convolution_type,
                weight_mode=self.weight_mode
            )
        )
            
        # Output heads
        self.output_head = ScalarOutputHead(
            irreps_hidden, out_node_nf, hidden_dim=hidden_nf
        )

        self.coord_head = VectorOutputHead(
            irreps_hidden, hidden_dim=hidden_nf
        )

        self.to(self.device)

    def compute_edge_features(self, x, edge_index):

        src, dst = edge_index

        rel_pos = x[dst] - x[src]
        edge_length = torch.linalg.norm(rel_pos, dim=1, keepdim=True)

        # Normalized edge vectors
        edge_vec = rel_pos / (edge_length + 1e-8)

        edge_sh = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True,
            normalization='component'
        )

        edge_length_embedded = self.radial_basis(edge_length.squeeze(-1))

        return edge_sh, edge_length_embedded
    
    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, 
                update_coords_mask=None, batch_mask=None, edge_attr=None):
        """
        Forward pass
        
        Args:
            h: [N, in_node_nf] - node features
            x: [N, 3] - node coordinates
            edge_index: [2, E] - edge indices
            node_mask: [N, 1] - node mask
            edge_mask: [E, 1] - edge mask
            update_coords_mask: [N, 1] - mask for coordinate updates
            batch_mask: [N] - batch indices
            edge_attr: [E, in_edge_nf] - edge attributes
            
        Returns:
            h_out: [N, out_node_nf] - output node features
            x_out: [N, 3] - output coordinates
        """
        edge_sh, edge_length_embedded = self.compute_edge_features(x, edge_index)

        if edge_attr is not None:
            edge_features = torch.cat([edge_length_embedded, edge_attr], dim=-1)

        else:
            edge_features = edge_length_embedded

        # Embed input

        h_embedded = self.embedding(h)
        h_irrpes= self.scalar_to_irreps(h_embedded)

        for conv in self.conv_layers:
            h_irrpes = conv(
                h_irrpes, edge_index, edge_sh, edge_features,
                batch_mask
            )        
        
        h_out = self.output_head(h_irrpes)
        coord_update = self.coord_head(h_irrpes)

        if update_coords_mask is not None:
            coord_update = coord_update * update_coords_mask
        
        x_out = x + coord_update

        if node_mask is not None:
            h_out = h_out * node_mask
            x_out = x_out * node_mask

        return h_out, x_out