from multiprocessing.sharedctypes import Value
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn

from .transformer_core import TransformerContainer, get_sine_cosine_pos_emb
from .skeleton import skeleton_parts_ids
from .gcnutils import Graph

class MoViT(nn.Module):
	"""ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
		<https://arxiv.org/abs/2103.15691>
	Args:
		num_frames (int): Number of frames in the video.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to 12.
		dropout_p (float): Probability of dropout layer. Defaults to 0..
		tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
			Defaults to 'fact_encoder'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'fact_encoder', 'joint_space_time', 'divided_space_time'
	]

	def __init__(self,
				 num_frames,
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=4,
				 dropout_p=0.,
				 tube_size=2,
				 attention_type='fact_encoder',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 extend_strategy='temporal_avg',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 data_rep='cont_6d', 
				 dataset='kit',
				 body_repres='full',	# or 'bodyparts' - coarse-grained using body parts
				 use_time_padding_mask=False,
				 skeleton_connection_mask_type='spatial',
				 skeleton_walk_len=1,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')
		
		# num_frames = num_frames//tube_size
		self.num_frames = num_frames
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.copy_strategy = copy_strategy
		self.extend_strategy = extend_strategy
		self.tube_size = tube_size
		self.num_time_transformer_layers = 0
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token
		self.body_repres = body_repres

		self.skel_parts_ids = skeleton_parts_ids[dataset]
		if body_repres == 'full':
			num_patches = 21 if dataset == 'kit' else 22
		else:
			num_patches = 5
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention - Model 3
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims,
				operator_order=operator_order)

			transformer_layers = container
		elif self.attention_type == 'joint_space_time':
			# Joint Space Time Attention - Model 1
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims,
				operator_order=operator_order)
			
			transformer_layers = container
		else:
			# Divided Space Time Transformer Encoder - Model 2
			transformer_layers = nn.ModuleList([])
			self.num_time_transformer_layers = 4
			
			spatial_transformer = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims,
				operator_order=['self_attn','ffn'])
			
			temporal_transformer = TransformerContainer(
				num_transformer_layers=self.num_time_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims,
				operator_order=['self_attn','ffn'])

			transformer_layers.append(spatial_transformer)
			transformer_layers.append(temporal_transformer)
 
		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_enb
		if attention_type == 'fact_encoder':
			num_frames = num_frames + 1
			num_patches = num_patches + 1
			self.use_cls_token_temporal = False
		else:
			self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
			if self.use_cls_token_temporal:
				num_frames = num_frames + 1
			else:
				num_patches = num_patches + 1

		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
			self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
			self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		self.drop_after_time = nn.Dropout(p=dropout_p)

		f = 6 if data_rep == 'cont_6d' else 6+3
		if body_repres == 'body-parts':
			self.layer1_rarm_enc = nn.Linear(f * len(self.skel_parts_ids['right_arm']), embed_dims)
			self.layer1_larm_enc = nn.Linear(f * len(self.skel_parts_ids['left_arm']), embed_dims)
			self.layer1_rleg_enc = nn.Linear(f * len(self.skel_parts_ids['right_leg']), embed_dims)
			self.layer1_lleg_enc = nn.Linear(f * len(self.skel_parts_ids['left_leg']), embed_dims)
			self.layer1_torso_enc = nn.Linear(f * len(self.skel_parts_ids['mid_body']), embed_dims)
		elif body_repres == 'full':
			self.body_enc = nn.Linear(f, embed_dims)
		else:
			raise ValueError(f'Body representation {body_repres} not known')

		# Handle temporal and position masks
		self.use_time_padding_mask = use_time_padding_mask
		self.skeleton_connection_mask_type = skeleton_connection_mask_type

		if skeleton_connection_mask_type is not None and body_repres != 'body-parts':
			graph_cfg=dict(layout=dataset, mode=skeleton_connection_mask_type, num_filter=8, init_off=.04, init_std=.02)
			graph = Graph(**graph_cfg)
			graph_adj = graph.A.max(0) != 0

			# add cls (collects infos from all the other joints, so first and last column are set to all ones)
			graph_adj_cls = torch.ones(graph_adj.shape[0] + 1, graph_adj.shape[1] + 1).bool()
			graph_adj_cls[1:, 1:] = torch.BoolTensor(graph_adj)

			self.graph_adj = graph_adj_cls
		else:
			self.skeleton_connection_mask_type = None

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def prepare_tokens(self, x):
		#Tokenize
		b = x.shape[0]
		x = rearrange(x, 'b t p d -> (b t) p d')
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
		x = self.drop_after_pos(x)

		# Add Time Embedding
		if self.attention_type != 'fact_encoder':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens,
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
		
		return x, cls_tokens, b

	def forward(self, x, lengths):

		x = x[:, :self.num_frames, ...]

		if self.use_time_padding_mask:
			bs = x.shape[0]
			temporal_mask = torch.zeros(bs, self.num_frames).bool()
			temporal_mask = temporal_mask.to(x.device)
			for m, c_len in zip(temporal_mask, lengths):
				m[c_len:] = True
		else:
			temporal_mask = None			

		if self.skeleton_connection_mask_type is not None:
			skeleton_att_mask = ~self.graph_adj.to(x.device)
		else:
			skeleton_att_mask = None

		# convert the input motion in a video-like input
		if self.body_repres == 'body-parts':
			b, t = x.shape[:2]
			right_arm = x[..., self.skel_parts_ids['right_arm'], :].view(b, t, -1)
			left_arm = x[..., self.skel_parts_ids['left_arm'], :].view(b, t, -1)
			right_leg = x[..., self.skel_parts_ids['right_leg'], :].view(b, t, -1)
			left_leg = x[..., self.skel_parts_ids['left_leg'], :].view(b, t, -1)
			mid_body = x[..., self.skel_parts_ids['mid_body'], :].view(b, t, -1)

			right_arm_layer1 = self.layer1_rarm_enc(right_arm)
			left_arm_layer1 = self.layer1_larm_enc(left_arm)
			mid_body_layer1 = self.layer1_torso_enc(mid_body)
			right_leg_layer1 = self.layer1_rleg_enc(right_leg)
			left_leg_layer1 = self.layer1_lleg_enc(left_leg)	# b x t x dim

			x = torch.stack([right_arm_layer1, left_arm_layer1, mid_body_layer1, right_leg_layer1, left_leg_layer1], dim=2)
		else:
			x = self.body_enc(x)

		# x is (batch x time x patches x dim)
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x, temporal_mask=temporal_mask, skeleton_mask=skeleton_att_mask)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x, attention_weights=skeleton_att_mask)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			
			x = temporal_transformer(x, padding_mask=temporal_mask)

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)

	def get_output_dim(self):
		return self.embed_dims

	def get_last_selfattention(self, x):
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x, return_attention=True)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			print(x.shape)
			x = temporal_transformer(x, return_attention=True)
		return x