# Loss Tracking During Training

To track per-sample losses during training with minimal overhead, you can modify the training loop as follows:


```
criterion = nn.CrossEntropyLoss()
<...>
loss = criterion(outputs, targets)
# here the loss is averaged out and
# does not have a batch dimension
loss.backward()
<...>
```

Here the loss reduction is set to `mean` by default and the loss is averaged over a batch before being returned 
from the `criterion` call. 
Instead, we can easily perform the aggregation on the client side, storing the per-sample losses before averaging.

```
criterion = nn.CrossEntropyLoss(
reduction="none"
)
<...>
loss = criterion(outputs, targets)
# here the loss is per-sample loss
# and does have a batch dimension
saved_losses.append(
loss.detach()
)
# take the mean before backward pass
loss.mean().backward()
<...>
```

We note, however, that this approach is not applicable if augmentations are used. In this case tracking per-epoch losses incurs
the cost of one forward pass on the full dataset on every epoch.

Memorization is indeed a property of a specific sample: the model’s behaviour on memorized samples and their augmented
versions might typically be very different. As such, when augmentations are applied during training, losses computed during
training would be losses on augmented samples, not the original ones. In this scenario we suggest recording the losses of the
non-augmented training samples in a separate forward pass after each epoch. While slightly more costly, this approach remains
computationally much cheaper than training even one shadow model, as a forward pass is typically several times cheaper than
the backward pass.