"""
Deep learning models of cortico-thalamo-cortical circuits built with the goal of elucidating the role of projections from high order thalamus to cortex.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class CortexWithThalamicMultiPreAct(nn.Module):
    """
    A custom layer which incorporates thalamic projection weightds applied after summation but before activation function.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)  # Standard linear layer
        self.relu = nn.ReLU()

    def forward(self, x, weights):
        """
        x: Input tensor of shape (batch_size, input_size)
        weights: Additional weight tensor of shape (batch_size, output_size)
        """
        linear_sum = self.linear(x)  # linear sum
        linear_weighted = linear_sum * weights  # element-wise multiplication with additional weights
        output = self.relu(linear_weighted)
        return output
    
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
                thalamocortical_type=None, # None, add (additive pre-sum), multi_pre_sum (multiplicative pre-sum), multi_pre_activation (multiplicative pre-activation), multi_post_activation (multiplicative post-activation)
                thal_reciprocal=True, # True or False
                thal_to_readout=True, # True or False
                thal_per_layer=False): # if no, mixing from cortical layers # determines spatial precision of connections and degree of spatial mixing
        super().__init__()
        
        # assign constructor args to class attributes
        self.input_size = input_size
        self.output_size = output_size
        self.ctx_layer_size = ctx_layer_size
        self.thal_layer_size = thal_layer_size
        self.thalamocortical_type = thalamocortical_type
        self.thal_reciprocal = thal_reciprocal
        self.thal_to_readout = thal_to_readout
        self.thal_per_layer = thal_per_layer

        # Thalamic layers
        # single thalamic area
        if not self.thal_per_layer:

            self.thal = nn.Sequential(
                nn.Linear(2 * self.ctx_layer_size, self.thal_layer_size),
                nn.ReLU()
            )

        # thalamic area per cortical area
        else:
            
            self.thal1 = nn.Sequential(
                nn.Linear(self.ctx_layer_size, self.thal_layer_size),
                nn.ReLU()
            )

            self.thal2 = nn.Sequential(
                nn.Linear(self.ctx_layer_size, self.thal_layer_size),
                nn.ReLU()
            )

        # thalamic feedback layers (required  to create layers for these to handle weights)
        if self.thalamocortical_type is not None:
            
            # to cortical layers
            if self.thal_reciprocal:
                if self.thalamocortical_type in ["multi_post_activation", "multi_pre_activation"]:
                    self.thal_to_ctx1_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                    self.thal_to_ctx2_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                else:
                    self.thal_to_ctx1_projections = nn.Linear(self.thal_layer_size, self.input_size)
                    self.thal_to_ctx2_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

            # to readout layer
            if self.thal_to_readout:
                
                # single thalamic area
                if not self.thal_per_layer:

                    if self.thalamocortical_type in ["multi_post_activation", "multi_pre_activation"]:
                        self.thal_to_readout_projections = nn.Linear(self.thal_layer_size, self.output_size)
                    else:
                        self.thal_to_readout_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

                # thalamic area per cortical area
                else:

                    if self.thalamocortical_type in ["multi_post_activation", "multi_pre_activation"]:
                        self.thal1_to_readout_projections = nn.Linear(self.thal_layer_size, self.output_size)
                        self.thal2_to_readout_projections = nn.Linear(self.thal_layer_size, self.output_size)
                    else:
                        self.thal1_to_readout_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                        self.thal2_to_readout_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

        # Cortical layers
        # special case for multiplicative pre-activation attention        
        if (self.thalamocortical_type == "multi_pre_activation") and self.thal_reciprocal:
            self.ctx1 = CortexWithThalamicMultiPreAct(self.input_size, self.ctx_layer_size)
            self.ctx2 = CortexWithThalamicMultiPreAct(self.ctx_layer_size, self.ctx_layer_size)
        else:
            self.ctx1 = nn.Sequential(
            nn.Linear(self.input_size, self.ctx_layer_size),
            nn.ReLU())
            self.ctx2 = nn.Sequential(
            nn.Linear(self.ctx_layer_size, self.ctx_layer_size),
            nn.ReLU()
        )

        # Readout layer
        self.readout = nn.Sequential(
            nn.Linear(self.ctx_layer_size, self.output_size),
            # nn.Softmax() # not required if using torch's cross entropy loss (since this applies softmax for you)
        )

    def forward(self, input):
        
        # flatten input (row-major)
        input = input.view(input.size(0), -1) # reshape input to flatten and remove second dimension (using view rather than assigning more memory)

        # one iteration of forward subroutine to get thalamic activity
        _, thal = self.subforward(input)

        # second iteration of forward subroutine to get output with forward activity in
        output, _ = self.subforward(input, thal=thal)

        return output

    def subforward(self, input, thal=None):

        # thalamocortical projections
        if self.thalamocortical_type is not None:

            # thalamic area per cortical area
            if self.thal_per_layer:

                # initialise thalamic activity with zeros
                if thal is None:
                    thal1 = torch.zeros([input.size(0), self.thal_layer_size], device=input.device)
                    thal2 = torch.zeros([input.size(0), self.thal_layer_size], device=input.device)     
                else:
                    thal1, thal2 = thal

                # compute thalamic feedback projection activity and subsequent cortical activity
                if self.thal_reciprocal:
                    thal_to_ctx1 = self.thal_to_ctx1_projections(thal1)
                    thal_to_ctx2 = self.thal_to_ctx2_projections(thal2)
                    if self.thalamocortical_type == "add":
                        ctx1 = self.ctx1(input + thal_to_ctx1)
                        ctx2 = self.ctx2(ctx1 + thal_to_ctx2 )
                    elif self.thalamocortical_type == "multi_pre_sum":
                        ctx1 = self.ctx1(input * thal_to_ctx1)
                        ctx2 = self.ctx2(ctx1 * thal_to_ctx2 )
                    elif self.thalamocortical_type == "multi_pre_activation":
                        ctx1 = self.ctx1(input, thal_to_ctx1)
                        ctx2 = self.ctx2(ctx1, thal_to_ctx2)
                    elif self.thalamocortical_type == "multi_post_activation":
                        ctx1 = self.ctx1(input) * thal_to_ctx1
                        ctx2 = self.ctx2(ctx1) * thal_to_ctx2
                # handle case for when readout feedback only    
                else: 
                    ctx1 = self.ctx1(input)
                    ctx2 = self.ctx2(ctx1)
            
                # compute thalamic feedback projection activity to readout and subsequent readout activity
                if self.thal_to_readout:
                    thal1_to_readout = self.thal1_to_readout_projections(thal1)
                    thal2_to_readout = self.thal2_to_readout_projections(thal2)
                    if self.thalamocortical_type == "add":
                        output = self.readout(ctx2 + thal1_to_readout + thal2_to_readout)
                    elif self.thalamocortical_type == "multi_pre_sum":
                        output = self.readout(ctx2 * thal1_to_readout * thal2_to_readout)
                    elif self.thalamocortical_type in ["multi_post_activation", "multi_pre_activation"]:
                        output = self.readout(ctx2) * thal1_to_readout * thal2_to_readout
                # handle case for when reciprocal feedback but no readout feedback 
                else: 
                    output = self.readout(ctx2)

            # single thalamic area 
            else:
            
                # initialise thalamic activity with zeros
                if thal is None:
                    thal = torch.zeros([input.size(0), self.thal_layer_size], device=input.device)

                # compute thalamic feedback projection activity and subsequent cortical activity
                if self.thal_reciprocal:
                    thal_to_ctx1 = self.thal_to_ctx1_projections(thal)
                    thal_to_ctx2 = self.thal_to_ctx2_projections(thal)
                    if self.thalamocortical_type == "add":
                        ctx1 = self.ctx1(input + thal_to_ctx1)
                        ctx2 = self.ctx2(ctx1 + thal_to_ctx2 )
                    elif self.thalamocortical_type == "multi_pre_sum":
                        ctx1 = self.ctx1(input * thal_to_ctx1)
                        ctx2 = self.ctx2(ctx1 * thal_to_ctx2 )
                    elif self.thalamocortical_type == "multi_pre_activation":
                        ctx1 = self.ctx1(input, thal_to_ctx1)
                        ctx2 = self.ctx2(ctx1, thal_to_ctx2)
                    elif self.thalamocortical_type == "multi_post_activation":
                        ctx1 = self.ctx1(input) * thal_to_ctx1
                        ctx2 = self.ctx2(ctx1) * thal_to_ctx2
                # handle case for when readout feedback only    
                else: 
                    ctx1 = self.ctx1(input)
                    ctx2 = self.ctx2(ctx1)

                # compute thalamic feedback projection activity to readout and subsequent readout activity
                if self.thal_to_readout:
                    thal_to_readout = self.thal_to_readout_projections(thal)
                    if self.thalamocortical_type == "add":
                        output = self.readout(ctx2 + thal_to_readout)
                    elif self.thalamocortical_type == "multi_pre_sum":
                        output = self.readout(ctx2 * thal_to_readout)
                    elif self.thalamocortical_type in ["multi_post_activation", "multi_pre_activation"]:
                        output = self.readout(ctx2) * thal_to_readout
               # handle case for when reciprocal feedback but no readout feedback 
                else: 
                    output = self.readout(ctx2)
            
            # compute thalamic activity for next timestep
            if self.thal_per_layer:
                thal1 = self.thal1(ctx1)
                thal2 = self.thal1(ctx2)
                thal = [thal1, thal2]
            else:
                input_thal = torch.cat([ctx1, ctx2], axis=1)
                thal = self.thal(input_thal)

        # no thalamocortical projections (purely feedforward)
        else:
            # compute cortical activity
            ctx1 = self.ctx1(input)
            ctx2 = self.ctx2(ctx1)

            # compute readout activity 
            output = self.readout(ctx2)

        return output, thal
        
    def summary(self):
        summary(self)

class CTCNetPlasticityMod(nn.Module):
    """
    Class for models which use thalamic layers for plasticity modulation.
    In these models, thalamic activity will be computed, but there will be no
    explicit thalamocortical projections. Instead, these will be fed into the 
    
    NOTE: forward function must output thalamic states
    """
    
    def forward():
        y_est = 0
        thal = 0
        return y_est, thal