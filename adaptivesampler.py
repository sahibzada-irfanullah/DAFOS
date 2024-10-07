from dgl.dataloading import MultiLayerNeighborSampler
# Adaptive Neighbor Sampler
class AdaptiveNeighborSampler:
    def __init__(self, threshold):
        self.threshold = threshold

    def update_fanout(self, current_fanout, loss_diff):
        # Dynamically increase fanout if loss difference is small (indicating a plateau)
        if loss_diff < self.threshold:
            current_fanout += 5  # Increase fanout by 5 when loss plateaus
        return current_fanout

    def get_sampler(self, current_fanout):
        return MultiLayerNeighborSampler([current_fanout[i] for i in range(2)])


