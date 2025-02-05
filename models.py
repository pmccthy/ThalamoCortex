"""
Deep learning models of cortico-thalamo-cortical circuits.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
# TODO: import BurstCCN here

# Set backend
print("Setting backend.")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device.")

# class FF(nn.Module):
#     """
#     Feedforward neural network.
#     TODO: make number of layers configurable
#     """
#     def __init__(self, self.input_size, self.input_size, layer_size, topographic=False, fan_in=None): 
#         self.super.__init__()

#         if not topographic:

#             self.layer1 = nn.Sequential(
#                 nn.Linear(self.input_size, layer_size),
#                 nn.ReLU())
            
#             self.layer2 = nn.Sequential(
#                 nn.Linear(layer_size, layer_size),
#                 nn.ReLU()
#             )

#             self.output = nn.Sequential(
#                 nn.Linear(layer_size, self.input_size),
#                 nn.Softmax()
#             )

#         elif topographic:
#             pass

#     def forward(self, input):
#         output1 = self.layer1(input)
#         output2 = self.layer2(output1)
#         output = self.layer3(output2)
#         return output

# class FFGlobalPlasticityModulation:
#     """
#     Feedforward neural network with input-dependent learning rate.
#     Biological analogue: cortical pathway with learning rate controlled by external region (assumed to be HoT).
#     """
#     pass

# # TODO: come up with some better terminology for referring to "cortical" and "thalamic" layers that is understandable in ML terms.

# class FFTransthalamicSeparate:
#     """
#     Feedforward neural network with every layer projecting to next layer as well as to another 
#     layer which itself projects directly to the output layer. Each layer has its own additional layer
#     providing another pathway to the output layer, and inputs from other layers are not mixed.
#     Biological analogue: transthalamic pathway to high cortical areas enabling more direct propagation of error/gradient information to lower cortical areas.
#     TODO: make number of layers configurable
#     """
#     def __init__(self, self.input_size, self.input_size, self.ctx_layer_size, self.thal_layer_size, topographic=False, fan_in=None): 
#         self.super.__init__()

#         if not topographic:

#             self.ctx1 = nn.Sequential(
#                 nn.Linear(self.input_size, self.ctx_layer_size),
#                 nn.ReLU())
        
#             self.thal1 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             self.ctx2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.ctx_layer_size),
#                 nn.ReLU()
#             )

#             self.thal2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             self.output = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size + 2 * self.thal_layer_size, self.input_size),
#                 nn.Softmax()
#             )

#         elif topographic:
#             pass
    
#         def forward(self, input):
#             ctx1 = self.ctx1(input)
#             thal1 = self.thal1(ctx1)
#             ctx2 = self.ctx2(ctx1)
#             thal2 = self.thal2(ctx2)
#             inputs_output_layer = torch.cat([ctx2, thal1, thal2])
#             output = self.ctx3(inputs_output_layer, self.input_size)
#             return output

# class FFTransthalamicMixed:
#     """
#     Feedforward neural network with every layer projecting to next layer as well as to another 
#     layer which itself projects directly to the output layer. Inputs from all layers are mixed
#     in the additional layer.
#     Biological analogue: transthalamic pathway to high cortical areas enabling more direct propagation of error/gradient information to lower cortical areas,
#     plus mixing of information from different layers.
#     """
#     def __init__(self, self.input_size, self.input_size, self.ctx_layer_size, self.thal_layer_size, topographic=False, fan_in=None): 
#         self.super.__init__()

#         if not topographic:

#             self.ctx1 = nn.Sequential(
#                 nn.Linear(self.input_size, self.ctx_layer_size),
#                 nn.ReLU())
        

#             self.ctx2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.ctx_layer_size),
#                 nn.ReLU()
#             )

#             self.thal = nn.Sequential(
#                 nn.Linear(2 * self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             self.output = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size + self.thal_layer_size, self.input_size),
#                 nn.Softmax()
#             )

#         elif topographic:
#             pass
    
#         def forward(self, input):
#             ctx1 = self.ctx1(input)
#             ctx2 = self.ctx2(ctx1)
#             input_thal = torch.cat([ctx1, ctx2])
#             thal = self.thal(input_thal)
#             inputs_output_layer = torch.cat([ctx2, thal])
#             output = self.ctx3(inputs_output_layer, self.input_size)
#             return output

# class FFTransthalamicReciprocalSeparate:
#     """
#     Feedforward neural network with every layer projecting to next layer as well as to another 
#     layer which itself projects directly to the output layer. Each layer has its own additional layer
#     providing another pathway to the output layer, and inputs from other layers are not mixed.
#     Each thalamic area also projects back to the cortical area it receives connections from.
#     Biological analogue: Not sure on biological interpretation of projections back to cortical areas. Perhaps reinforce representations.
#     NOTE: thalamic projection will be delayed by one timestep so input must be formulated as sequence.
#     """
#     def __init__(self, self.input_size, self.input_size, self.ctx_layer_size, self.thal_layer_size, topographic=False, fan_in=None): 
#         self.super.__init__()

#         if not topographic:
            
#             # feedforward cortical 1
#             self.ctx1 = nn.Sequential(
#                 nn.Linear(self.input_size, self.ctx_layer_size),
#                 nn.ReLU())

#             # corticothalamic 1
#             self.thal1 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             # feedforward cortical 2
#             self.ctx2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.ctx_layer_size),
#                 nn.ReLU()
#             )
            
#             # corticothalamic 2
#             self.thal2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             # thalamocortical feedback to cortical 1
#             self.thal1_to_ctx1 = nn.Linear(self.thal_layer_size, self.input_size)

#             # thalamocortical feedback to cortical 2
#             self.thal2_to_ctx2 = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

#             # thalamocortical feedback to cortical 1
#             self.thal1_to_output = nn.Linear(self.thal_layer_size, self.input_size)

#             # thalamocortical feedback to cortical 2
#             self.thal2_to_output = nn.Linear(self.thal_layer_size, self.input_size)

#             # readout layer
#             self.output = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size + 2 * self.thal_layer_size, self.input_size),
#                 nn.Softmax()
#             )

#         elif topographic:
#             pass
    
#         def forward(self, input, thal1=None, thal2=None):
#             # on first timestep, initialise thalamic layer activity with zeros
#             if thal1 is None:
#                 thal1 = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)
#             if thal2 is None:
#                 thal2 = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)
            
#             # compute thalamic feedback projection activity
#             thal1_to_ctx1 = self.thal_to_ctx1(thal1)
#             thal2_to_ctx2 = self.thal_to_ctx2(thal2)
#             thal1_to_output = self.thal1_to_output(thal1)
#             thal2_to_output = self.thal2_to_output(thal2)

#             # compute cortical activity with feedforward input and thalamic feedback
#             ctx1 = self.ctx1(input + thal1_to_ctx1) 
#             ctx2 = self.ctx2(ctx1 + thal2_to_ctx2)

#             # compute thalamic activity (thalamic feedback for next timestep)
#             thal1 = self.thal1(ctx1)
#             thal2 = self.thal2(ctx2)
            
#             output = self.ctx3(ctx2 + thal1_to_output + thal2_to_output, self.input_size)

#             return output, thal1, thal2
    

# class FFTransthalamicReciprocalMixed:
#     """
#     Feedforward neural network with every layer projecting to next layer as well as to another 
#     layer which itself projects directly to the output layer. Inputs from all layers are mixed
#     in the additional layer. Each thalamic area also projects back to the cortical area it receives connections from.
#     Biological analogue: Not sure on biological interpretation of projections back to cortical areas. Perhaps reinforce representations.
#     # NOTE: could rename this AdditiveFeedback to contrast with MultiplicativeFeedback classes below.
#     """
#     def __init__(self, self.input_size, self.input_size, self.ctx_layer_size, self.thal_layer_size, topographic=False, fan_in=None): 
#         self.super.__init__()

#         if not topographic:
            
#             # feedforward cortical 1
#             self.ctx1 = nn.Sequential(
#                 nn.Linear(self.input_size + self.thal_layer_size, self.ctx_layer_size),
#                 nn.ReLU())
        
#             # feedforward cortical 2
#             self.ctx2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size + self.thal_layer_size, self.ctx_layer_size),
#                 nn.ReLU()
#             )

#             # corticothalamic
#             self.thal = nn.Sequential(
#                 nn.Linear(2 * self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             # thalamocortical feedback to cortical 1
#             self.thal_to_ctx1 = nn.Linear(self.thal_layer_size, self.input_size)

#             # thalamocortical feedback to cortical 2
#             self.thal_to_ctx2 = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

#             # thalamocortical feedback to output layer
#             self.thal_to_output = nn.Linear(self.thal_layer_size, self.input_size)

#             # readout layer
#             self.output = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.input_size),
#                 nn.Softmax()
#             )

#         elif topographic:
#             pass
    
#         def forward(self, input, thal=None):

#             # on first timestep, initialise thalamic layer activity with zeros
#             if thal is None:
#                 thal = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)

#             # compute thalamic feedback projection activity
#             thal_to_ctx1 = self.thal_to_ctx1(thal)
#             thal_to_ctx2 = self.thal_to_ctx2(thal)
#             thal_to_output = self.thal_to_output(thal)

#             # compute cortical activity with feedforward input and thalamic feedback
#             ctx1 = self.ctx1(input + thal_to_ctx1)
#             ctx2 = self.ctx2(ctx1 + thal_to_ctx2)

#             # compute thalamic layer activity (will determine thalamic feedback for next timestep)
#             input_thal = torch.cat([ctx1, ctx2])
#             thal = self.thal(input_thal)
            
#             # compute readout activity 
#             output = self.ctx3(ctx2 + thal_to_output, self.input_size)

#             return output, thal
        
# class FFMultiplicativeAttentionSeparate:
#     """
#     Feedforward neural network with every layer projecting to next layer as well as to another 
#     layer which feeds back to each layer with a weighting, serving as an attention mechanism.
#     Attention weights for each layer are computed separately.
#     Biological analogue: Attention mechanism where attention weights are learned separately for each layer.
#     """
#     def __init__(self, self.input_size, self.input_size, self.ctx_layer_size, self.thal_layer_size, topographic=False, fan_in=None): 
#         self.super.__init__()

#         if not topographic:
            
#             # feedforward cortical 1
#             self.ctx1 = nn.Sequential(
#                 nn.Linear(self.input_size, self.ctx_layer_size),
#                 nn.ReLU())

#             # corticothalamic 1
#             self.thal1 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             # feedforward cortical 2
#             self.ctx2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.ctx_layer_size),
#                 nn.ReLU()
#             )
            
#             # corticothalamic 2
#             self.thal2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             # thalamocortical feedback to cortical 1
#             self.thal1_to_ctx1 = nn.Linear(self.thal_layer_size, self.input_size)

#             # thalamocortical feedback to cortical 2
#             self.thal2_to_ctx2 = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

#             # readout layer
#             self.output = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size + 2 * self.thal_layer_size, self.input_size),
#                 nn.Softmax()
#             )

#         elif topographic:
#             pass
    
#         def forward(self, input, thal1=None, thal2=None):
#             # on first timestep, initialise thalamic layer activity with zeros
#             if thal1 is None:
#                 thal1 = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)
#             if thal2 is None:
#                 thal2 = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)
            
#             # compute thalamic feedback projection activity
#             thal1_to_ctx1 = self.thal_to_ctx1(thal1)
#             thal2_to_ctx2 = self.thal_to_ctx2(thal2)

#             # compute cortical activity with feedforward input and thalamic feedback
#             ctx1 = self.ctx1(input * thal1_to_ctx1) 
#             ctx2 = self.ctx2(ctx1 * thal2_to_ctx2)

#             # compute thalamic activity (thalamic feedback for next timestep)
#             thal1 = self.thal1(ctx1)
#             thal2 = self.thal2(ctx2)
            
#             output = self.ctx3(ctx2, self.input_size)

#             return output, thal1, thal2
    
# class FFMultiplicativeAttentionMixed:
#     """
#     Feedforward neural network with every layer projecting to next layer as well as to another 
#     layer which feeds back to each layer with a weighting, serving as an attention mechanism.
#     Attention weights for each layer are computed separately.
#     Biological analogue: Attention mechanism where attention weights are mixed across layers.
#     """
#     def __init__(self, self.input_size, self.input_size, self.ctx_layer_size, self.thal_layer_size, topographic=False, fan_in=None): 
#         self.super.__init__()

#         if not topographic:
            
#             # feedforward cortical 1
#             self.ctx1 = nn.Sequential(
#                 nn.Linear(self.input_size + self.thal_layer_size, self.ctx_layer_size),
#                 nn.ReLU())
        
#             # feedforward cortical 2
#             self.ctx2 = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size + self.thal_layer_size, self.ctx_layer_size),
#                 nn.ReLU()
#             )

#             # corticothalamic
#             self.thal = nn.Sequential(
#                 nn.Linear(2 * self.ctx_layer_size, self.thal_layer_size),
#                 nn.ReLU()
#             )

#             # thalamocortical feedback to cortical 1
#             self.thal_to_ctx1 = nn.Linear(self.thal_layer_size, self.input_size)

#             # thalamocortical feedback to cortical 2
#             self.thal_to_ctx2 = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

#             # thalamocortical feedback to output layer
#             self.thal_to_output = nn.Linear(self.thal_layer_size, self.input_size)

#             # readout layer
#             self.output = nn.Sequential(
#                 nn.Linear(self.ctx_layer_size, self.input_size),
#                 nn.Softmax()
#             )

#         elif topographic:
#             pass
    
#         def forward(self, input, thal=None):

#             # on first timestep, initialise thalamic layer activity with zeros
#             if thal is None:
#                 thal = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)

#             # compute thalamic feedback projection activity
#             thal_to_ctx1 = self.thal_to_ctx1(thal)
#             thal_to_ctx2 = self.thal_to_ctx2(thal)
#             thal_to_output = self.thal_to_output(thal)

#             # compute cortical activity with feedforward input and thalamic feedback
#             ctx1 = self.ctx1(input * thal_to_ctx1)
#             ctx2 = self.ctx2(ctx1 * thal_to_ctx2)

#             # compute thalamic layer activity (will determine thalamic feedback for next timestep)
#             input_thal = torch.cat([ctx1, ctx2])
#             thal = self.thal(input_thal)
            
#             # compute readout activity 
#             output = self.ctx3(ctx2 + thal_to_output, self.input_size)

#             return output, thal
        
class CTCNet(nn.Module):
    """
    A unifying class enabling implementation of all of the above models through specific parameter choices.
    # TODO: implement option for turning off gradients for specific connections (analagous to removing error feedback projections)
    # TODO: extend to allow for any number of cortical and thalamic layers (when one-to-one correspondence)
    # TODO: add dropout layers with configurable sparsity
   """
    def __init__(self,
                input_size,
                output_size,
                ctx_layer_size,
                thal_layer_size,
                thalamocortical_type=None, # None, multiplicative, or additive
                thal_reciprocal=True, # True or False
                thal_to_readout=True, # True or False
                thal_per_layer=False): # if no, mixing from cortical layers # determines spatial precision of connections and degree of spatial mixing
        super().__init__()
        
        # assign class variables
        self.input_size = input_size
        self.output_size = output_size
        self.ctx_layer_size = ctx_layer_size
        self.thal_layer_size = thal_layer_size
        self.thalamocortical_type = thalamocortical_type
        self.thal_reciprocal = thal_reciprocal
        self.thal_to_readout = thal_to_readout
        self.thal_per_layer = thal_per_layer

        if not self.thal_per_layer:

            # corticothalamic
            self.thal = nn.Sequential(
                nn.Linear(2 * self.ctx_layer_size, self.thal_layer_size),
                nn.ReLU()
            )

        else:
            
            # corticothalamic
            self.thal1 = nn.Sequential(
                nn.Linear(self.ctx_layer_size, self.thal_layer_size),
                nn.ReLU()
            )

          # corticothalamic
            self.thal2 = nn.Sequential(
                nn.Linear(self.ctx_layer_size, self.thal_layer_size),
                nn.ReLU()
            )

        # compute thalamocortical feedback activity 
        if self.thalamocortical_type is not None:
            if self.thal_reciprocal:
                if self.thalamocortical_type == "multi_post_activation":
                    self.thal_to_ctx1 = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                    self.thal_to_ctx2 = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                else:
                    self.thal_to_ctx1 = nn.Linear(self.thal_layer_size, self.input_size)
                    self.thal_to_ctx2 = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

            if self.thal_to_readout:

                if not self.thal_per_layer:

                    if self.thalamocortical_type == "multi_post_activation":
                        self.thal_to_readout = nn.Linear(self.thal_layer_size, self.input_size)
                    else:
                        self.thal_to_readout = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

                else:

                    if self.thalamocortical_type == "multi_post_activation":
                        self.thal1_to_readout = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                        self.thal2_to_readout = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                    else:
                        self.thal1_to_readout = nn.Linear(self.thal_layer_size, self.input_size)
                        self.thal2_to_readout = nn.Linear(self.thal_layer_size, self.input_size)

        # feedforward cortical 1
        self.ctx1 = nn.Sequential(
            nn.Linear(self.input_size, self.ctx_layer_size),
            nn.ReLU())
    
        # feedforward cortical 2
        # TODO: add new arguments to these layers to accommodate attention weights and attention type
        #       to allow more easy implementation of four types of attention
        self.ctx2 = nn.Sequential(
            nn.Linear(self.ctx_layer_size, self.ctx_layer_size),
            nn.ReLU()
        )
        
        # readout layer
        self.readout = nn.Sequential(
            nn.Linear(self.ctx_layer_size, self.input_size),
            nn.Softmax()
        )

    def forward(self, input):

        # one iteration of forward subroutine to get thalamic activity
        _, thal = self.subforward(input)

        # second iteration of forward subroutine to get output with forward activity in
        output, _ = self.subforward(input, thal=thal)

        return output

    def subforward(self, input, thal=None):

        if self.thalamocortical_type is not None:

            if self.thal_per_layer:

                # initialise thalamic activity with zeros
                if thal is None:
                    thal1 = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)
                    thal2 = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)     
                else:
                    thal1, thal2 = thal
                # compute thalamic feedback projection activity
                if self.thal_reciprocal:
                    thal_to_ctx1 = self.thal_to_ctx1(thal1)
                    thal_to_ctx2 = self.thal_to_ctx2(thal2)
                    if self.thalamocortical_type == "add":
                        input_ctx1 = input + thal_to_ctx1
                        input_ctx2 = ctx1 + thal_to_ctx2 
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_ctx1 = input * thal_to_ctx1
                        input_ctx2 = ctx1 * thal_to_ctx2 
                if self.thal_to_readout:
                    thal1_to_readout = self.thal1_to_readout(thal1)
                    thal2_to_readout = self.thal2_to_readout(thal2)
                    if self.thalamocortical_type == "add":
                        input_readout = ctx2 + thal1_to_readout + thal2_to_readout
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_readout = ctx2 * thal1_to_readout * thal2_to_readout
            else:
            
                # initialise thalamic activity with zeros
                if thal is None:
                    thal = torch.zeros(self.thal_layer_size.size(0), self.thal_layer_size, device=input.device)

                # compute thalamic feedback projection activity
                if self.thal_reciprocal:
                    thal_to_ctx1 = self.thal_to_ctx1(thal)
                    thal_to_ctx2 = self.thal_to_ctx2(thal)
                    if self.thalamocortical_type == "add":
                        input_ctx1 = input + thal_to_ctx1
                        input_ctx2 = ctx1 + thal_to_ctx2 
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_ctx1 = input * thal_to_ctx1
                        input_ctx2 = ctx1 * thal_to_ctx2 
                if self.thal_to_readout:
                    self.thal_to_readout = self.thal_to_readout(thal)
                    if self.thalamocortical_type == "add":
                        input_readout = ctx2 + self.thal_to_readout
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_readout = ctx2 * self.thal_to_readout
            
            # compute cortical activity
            # TODO: enable multiplicative attention after activations

            if self.thalamocortical_type == "multi_post_activation":
                ctx1 = self.ctx1(input_ctx1) * thal_to_ctx1
                ctx2 = self.ctx2(input_ctx2) * thal_to_ctx2

                # compute readout activity 
                if self.thal_per_layer:
                    output = self.readout(input_readout) * thal1_to_readout * thal2_to_readout
                else:
                    output = self.readout(input_readout) * self.thal_to_readout

            else:
                ctx1 = self.ctx1(input_ctx1)
                ctx2 = self.ctx2(input_ctx2)

                # compute readout activity 
                output = self.readout(input_readout)

            # compute thalamic activity for next timestep
            if self.thal_per_layer:
                thal1 = self.thal1(ctx1)
                thal2 = self.thal1(ctx2)
                thal = [thal1, thal2]
            else:
                input_thal = torch.cat([ctx1, ctx2])
                thal = self.thal(input_thal)

        else:
            # compute cortical activity
            ctx1 = self.ctx1(input)
            ctx2 = self.ctx2(ctx1)
            # compute readout activity 
            output = self.readout(ctx2)

        return output, thal
        
    def summary(self):
        summary(self)
        

def train_dynamic_learning_rate(data_loader, model, loss_fn, optimizer, class_subgroups={"group1": [1,2,3,4],
                                                                                         "group2": [5,6,7,8]}):
    """Train model for one epoch."""

    size = len(data_loader.dataset)
    model.train()
    losses = []
    for batch, (X, Y) in enumerate(data_loader):
        # move data to device where model will be trained
        X = X.to(device)

        # compute error
        Y, X_recon = model(X)
        loss = loss_fn(X_recon, X)

        # set learning rate based on class ID
        for group_name, group_vals in class_subgroups.items():
            if Y in group_vals:
                group = group_name
        optimizer = torch.optim.Adam(model.parameters(), lr=HP["LEARNING_RATE"][group_name])

        # backprop
        loss.backward()  # compute gradients
        optimizer.step()  # update params
        optimizer.zero_grad()  # ensure not tracking gradients fo next iteration

        # print loss every 500th batch
        if batch % HP["LOSS_TRACK_STEP"] == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"training batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
            )
            losses.append(loss)

    return losses
 
# def train_reciprocal(...):
#     pass

class WeightedFeedforwardLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu):
        super(WeightedFeedforwardLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = activation  # Activation function

    def forward(self, x, attention_weights):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).
            attention_weights (Tensor): Weighting tensor of shape (batch_size, output_dim),
                                        which scales the activations.
        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        out = self.linear(x)         # Linear transformation
        out = self.activation(out)   # Apply non-linearity
        out = out * attention_weights  # Element-wise multiplication with attention weights
        return out
