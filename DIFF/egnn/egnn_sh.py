import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import TensorProduct
from e3nn.nn import Gate, BatchNorm
from e3nn.o3 import Linear, FullyConnectedTensorProduct
from torch_scatter import scatter
import math
from mace.modules import (
    RealAgnosticInteractionBlock, 
    EquivariantProductBasisBlock)
from mace.modules.blocks import RealAgnosticResidualInteractionBlock 

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

class MACEBlock(nn.Module):

    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_hidden: o3.Irreps, 
        irreps_sh: o3.Irreps,
        max_correlation_order=3,  # 4-body interactions by default
        hidden_dim=64, 
        num_rbf: int = 16,
    ):
        super().__init__()
        self.norm = BatchNorm(irreps_node_hidden)
        self.max_correlation_order = max_correlation_order
        
        self.interaction = RealAgnosticResidualInteractionBlock(
            node_feats_irreps=irreps_node_input,
            target_irreps=irreps_node_hidden,
            hidden_irreps=irreps_node_hidden,
            avg_num_neighbors=1.0,
            edge_attrs_irreps=irreps_sh,
            edge_feats_irreps=o3.Irreps(f"{num_rbf}x0e"), 
            node_attrs_irreps=o3.Irreps("1x0e"),
            radial_MLP=[64, 64, 64]
        )
        
        # Bước 2: Khối Many-Body (Product Basis) để tạo message cuối cùng
        self.product_basis = EquivariantProductBasisBlock(
            node_feats_irreps=irreps_node_hidden, # Đầu vào là A-features
            target_irreps=irreps_node_hidden,      # Đầu ra là message
            correlation=max_correlation_order,
            use_sc=True,
            use_agnostic_product=True,
            num_elements=1
        )
    
    def forward(self, h, edge_index, edge_sh, edge_radial_embedding, batch_mask=None, cutoff=None,**kwargs):
        
        node_attrs_dummy = torch.ones(
            (h.shape[0], 1), 
            dtype=h.dtype, 
            device=h.device
        )
        # Bước 1: Tạo A-features từ tương tác 2-body
        a_features, sc = self.interaction(
            node_feats=h,
            node_attrs=node_attrs_dummy,
            edge_attrs=edge_sh,
            edge_feats=edge_radial_embedding,
            edge_index=edge_index,
            cutoff=cutoff
        )
        
        # Bước 2: Tạo message cuối cùng từ tương tác many-body
        messages = self.product_basis(
            node_feats=a_features,
            sc=sc, # Truyền skip connection từ bước tương tác
            node_attrs=node_attrs_dummy
        )
        
        h_out = self.norm(messages)
        return h_out
    
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
                    nn.Linear(hidden_dim, hidden_dim),
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
                 rbf="expnormal", trainable_rbf=True, 
                 block_type='conv', max_correlation_order =3,
                 convolution_type='separable', weight_mode='dynamic'):
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
        self.block_type = block_type
        
        """
                irreps_hidden = o3.Irreps(
            f"{hidden_nf//2}x0e + {hidden_nf//4}x1o + {hidden_nf//8}x2e"
        )
        """
        from mace.modules.radial import PolynomialCutoff
        self.cutoff_fn = PolynomialCutoff(r_max=max_radius, p=6)
        
        num_channels = hidden_nf // 2

        irreps_string = "+".join([f"{num_channels}x{l}{p}" for l in range(lmax + 1) for p in [-1, 1] if l % 2 == (p + 1) % 2])
        irreps_string = f"{num_channels}x0e + {num_channels}x1o + {num_channels}x2e"
        irreps_hidden = o3.Irreps(irreps_string)
        
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
        self.conv_layers = nn.ModuleList()

        current_irreps = self.scalar_to_irreps.irreps_out
        
        for i in range(n_layers):
            if self.block_type == 'mace':
                block = MACEBlock(
                    irreps_node_input=current_irreps,
                    irreps_node_hidden=irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    max_correlation_order=max_correlation_order,
                    hidden_dim=hidden_nf, num_rbf=self.num_rbf
                )                    
                current_irreps = irreps_hidden

            elif self.block_type == 'conv':
                block_irreps_in = current_irreps if i == 0 else irreps_hidden
                
                block = SphericalHarmonicsBlock(
                    irreps_hidden=block_irreps_in, # Sửa lại để linh hoạt hơn
                    irreps_sh=self.irreps_sh,
                    edge_features=self.edge_embed_dim,
                    hidden_dim=hidden_nf,
                    residual=True,
                    normalization=normalization,
                    conv_type=convolution_type,
                    weight_mode=weight_mode
                )
                # Đầu ra của SphericalHarmonicsBlock cũng là irreps_hidden
                current_irreps = irreps_hidden

            else:
                raise ValueError(f"Unknown block_type: {self.block_type}. Choose 'conv' or 'mace'.")
            
            self.conv_layers.append(block)
        # Output heads
        self.output_head = ScalarOutputHead(
            current_irreps, out_node_nf, hidden_dim=hidden_nf
        )

        self.coord_head = VectorOutputHead(
            current_irreps, hidden_dim=hidden_nf
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
        cutoff = self.cutoff_fn(edge_length)

        return edge_sh, edge_length_embedded, cutoff  
    
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
        edge_sh, edge_length_embedded, cutoff = self.compute_edge_features(x, edge_index)

        if edge_attr is not None:
            edge_features = torch.cat([edge_length_embedded, edge_attr], dim=-1)

        else:
            edge_features = edge_length_embedded

        # Embed input

        h_embedded = self.embedding(h)
        h_irrpes= self.scalar_to_irreps(h_embedded)

        for conv_layer in self.conv_layers:
            if isinstance(conv_layer, MACEBlock):
                # MACEBlock chỉ nhận radial embedding
                h_irrpes = conv_layer(
                    h=h_irrpes,
                    edge_index=edge_index,
                    edge_sh=edge_sh,
                    edge_radial_embedding=edge_length_embedded, # Dùng biến gốc
                    batch_mask=batch_mask,
                    cutoff=cutoff
                )
            else: 
                # Khối này nhận toàn bộ edge_features (bao gồm cả edge_attr)
                h_irrpes = conv_layer(
                    h=h_irrpes,
                    edge_index=edge_index,
                    edge_sh=edge_sh,
                    edge_features=edge_features, # Dùng biến đã kết hợp
                    batch_mask=batch_mask
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