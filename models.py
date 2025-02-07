"""
Deep learning models of cortico-thalamo-cortical circuits.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
# TODO: import BurstCCN here

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
        
        # assign constructor args to class attributes
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
                    self.thal_to_ctx1_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                    self.thal_to_ctx2_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                else:
                    self.thal_to_ctx1_projections = nn.Linear(self.thal_layer_size, self.input_size)
                    self.thal_to_ctx2_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

            if self.thal_to_readout:

                if not self.thal_per_layer:

                    if self.thalamocortical_type == "multi_post_activation":
                        self.thal_to_readout_projections = nn.Linear(self.thal_layer_size, self.output_size)
                    else:
                        self.thal_to_readout_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

                else:

                    if self.thalamocortical_type == "multi_post_activation":
                        self.thal1_to_readout_projections = nn.Linear(self.thal_layer_size, self.output_size)
                        self.thal2_to_readout_projections = nn.Linear(self.thal_layer_size, self.output_size)
                    else:
                        self.thal1_to_readout_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)
                        self.thal2_to_readout_projections = nn.Linear(self.thal_layer_size, self.ctx_layer_size)

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
            nn.Linear(self.ctx_layer_size, self.output_size),
            # nn.Softmax() # not required if using torch's cross entropy loss (since this applies softmax for you)
        )

    def forward(self, input):
        
        # flatten input
        input = input.view(input.size(0), -1) # reshape input to flatten and remove second dimension (using view rather than assigning more memory)

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
                    thal1 = torch.zeros([input.size(0), self.thal_layer_size], device=input.device)
                    thal2 = torch.zeros([input.size(0), self.thal_layer_size], device=input.device)     
                else:
                    thal1, thal2 = thal

                # compute thalamic feedback projection activity
                if self.thal_reciprocal:
                    thal_to_ctx1 = self.thal_to_ctx1_projections(thal1)
                    thal_to_ctx2 = self.thal_to_ctx2_projections(thal2)
                    if self.thalamocortical_type == "add":
                        input_ctx1 = input + thal_to_ctx1
                        ctx1 = self.ctx1(input_ctx1)
                        input_ctx2 = ctx1 + thal_to_ctx2 
                        ctx2 = self.ctx2(input_ctx2)
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_ctx1 = input * thal_to_ctx1
                        ctx1 = self.ctx1(input_ctx1)
                        input_ctx2 = ctx1 * thal_to_ctx2 
                        ctx2 = self.ctx2(input_ctx2)
                    elif self.thalamocortical_type == "multi_post_activation":
                        input_ctx1 = input
                        ctx1 = self.ctx1(input_ctx1) * thal_to_ctx1
                        input_ctx2 = ctx1
                        ctx2 = self.ctx2(input_ctx2) * thal_to_ctx2
                else: # handles case for when readout feedback only
                    ctx1 = self.ctx1(input)
                    ctx2 = self.ctx2(ctx1)
                              
                if self.thal_to_readout:
                    thal1_to_readout = self.thal1_to_readout_projections(thal1)
                    thal2_to_readout = self.thal2_to_readout_projections(thal2)
                    if self.thalamocortical_type == "add":
                        input_readout = ctx2 + thal1_to_readout + thal2_to_readout
                        output = self.readout(input_readout)
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_readout = ctx2 * thal1_to_readout * thal2_to_readout
                        output = self.readout(input_readout)
                    elif self.thalamocortical_type == "multi_post_activation":
                        input_readout = ctx2
                        output = self.readout(input_readout) * thal1_to_readout * thal2_to_readout
                else: # handles case for when reciprocal feedback but no readout feedback
                    output = self.readout(ctx2)

            else:
            
                # initialise thalamic activity with zeros
                if thal is None:
                    thal = torch.zeros([input.size(0), self.thal_layer_size], device=input.device)

                # compute thalamic feedback projection activity
                if self.thal_reciprocal:
                    thal_to_ctx1 = self.thal_to_ctx1_projections(thal)
                    thal_to_ctx2 = self.thal_to_ctx2_projections(thal)
                    if self.thalamocortical_type == "add":
                        input_ctx1 = input + thal_to_ctx1
                        ctx1 = self.ctx1(input_ctx1)
                        input_ctx2 = ctx1 + thal_to_ctx2 
                        ctx2 = self.ctx2(input_ctx2)
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_ctx1 = input * thal_to_ctx1
                        ctx1 = self.ctx1(input_ctx1)
                        input_ctx2 = ctx1 * thal_to_ctx2 
                        ctx2 = self.ctx2(input_ctx2)
                    elif self.thalamocortical_type == "multi_post_activation":
                        ctx1 = self.ctx1(input) * thal_to_ctx1
                        ctx2 = self.ctx2(ctx1) * thal_to_ctx2
                else: # handles case for when readout feedback only
                    ctx1 = self.ctx1(input)
                    ctx2 = self.ctx2(ctx1)

                if self.thal_to_readout:
                    thal_to_readout = self.thal_to_readout_projections(thal)
                    if self.thalamocortical_type == "add":
                        input_readout = ctx2 + thal_to_readout
                        output = self.readout(input_readout)
                    elif self.thalamocortical_type == "multi_pre_activation":
                        input_readout = ctx2 * thal_to_readout
                        output = self.readout(input_readout)
                    elif self.thalamocortical_type == "multi_post_activation":
                        input_readout = ctx2
                        output = self.readout(input_readout) * thal_to_readout
                else: # handles case for when reciprocal feedback but no readout feedback
                    output = self.readout(ctx2)

            
            # compute thalamic activity for next timestep
            if self.thal_per_layer:
                thal1 = self.thal1(ctx1)
                thal2 = self.thal1(ctx2)
                thal = [thal1, thal2]
            else:
                input_thal = torch.cat([ctx1, ctx2], axis=1)
                # print(f"{input_thal.shape=}")
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