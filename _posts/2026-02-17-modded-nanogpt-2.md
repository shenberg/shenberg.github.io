---
title: "Modded-NanoGPT part 2"
date: 2026-02-17 12:00:00 +0000
slug: modded-nanogpt-2
excerpt: ""
---

I just had the good fortune to slightly improve the [NanoGPT speedrun, modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) record, three separate times. This is part two, in which I talk about records 2 & 3.

The [first post](TODO) talks about modded-nanoGPT in general and is a better starting point.

The improvements were:
1. [Catious Weight Decay]() on Adam + tuning
2. Sparse all-to-all gradient communication for a parameter that is relatively sparse.

Cautious weight-decay was straightforward but left me with unanswered questions. Sparse all-to-all was an interesting engineering challenge that is still only half-finished.

<!--more-->

### Cautious Weight Decay

To explain catious weight decay, it's worthwhile to explain the evolution of L2 regularization -> decoupled weight-decay -> cautious weight decay.

All the methods try to enforce the idea that "small-magnitude parameters are more generalizable." How do we force parameters to be small while still solving the problem?

L2-regularization solved for the problem by adding an additonal loss term $\lambda \lVert \theta\rVert ^2 $ for some small $ \lambda > 0$. Under gradient descent, this works out to weight decay, which means that the update $\theta_{t+1} = (1-\lambda)\theta_{t} - \mu \nabla \mathfrak{L}$ for some learning rate $\mu$ and loss $\mathfrak{L}$. Weight-decay is this multiplication of the parameters of the model by a constant <1 every iteration of the optimization process.

Weight-decay proved to be unhelpful and unpopular in the first deep learning gold-rush after AlexNet. The authors of "[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)" found the problem and fixed it. They noted that L2 regularization and weight decay are not equivalent under the popular optimizers of the time, while deep learning libraries had a "weight decay" parameter, they actually implemented L2 regularization and modified the gradient.

AdamW, their variant of Adam, implemented weight decay by leaving the gradients alone and directly modifying the weights of the model every step, became the baseline optimizer roughly till today.

Catious weight decay goes a step further by limiting when weight decay is applied, only applying it if the decay sign (per parameter) agrees with the step direction of the optimizer. What the authors show is that while weight decay changes the optimization target from the original loss to some surrogate function that doesn't necessarily have the same minima, their formulation keeps the same convergence points. 

Catious weight decay was already [implemented by varunneal](TODO) for Muon, the optimizer for the MLP and self-attention matrices in modded-nanogpt. I implemented it in the customized Adam implementation used in modded-nanogpt.

The implementation was straightforward, the run-time cost was effectively zero (the optimizer is communication-bound between GPUs, not compute-bound, so there was almost no run-time overhead for calculations done while waiting for networking), and generalization improved after a bit of tuning, so less training steps were necessary to reach the loss threshold.

TODO: less-catious weight decay? removing weight decay? weight decay schedule?

## Sparse Communication
### Why do we need it?
A recent record run added a new bigram-embedding table, inspired by the  [deepseek engram paper](https://www.arxiv.org/abs/2601.07372). This added a new parameter, a hash embedding table with $\approx 250,000$ rows. Batch sizes per GPU range between 32,768 and 98,304 tokens, so even assuming a worst-case scenario of every bigram in the batch being mapped to a separate row (which is _very_ unlikely), only 40% of the table will be used in the largest batch size. (Note: Adam is stepped every other training step, so the sizes above are actually for pairs of batches)

In practice, looking at 50 batches, we get an average 21.5% occupancy (thank you birthday paradox). (side-note: the hash function in use is weird but is pretty close to ideal on the measured batches)

Ideally, this means we can send a lot less data - a fifth in the worst case. The rest is just implementation details.

The common way to share data across GPUs in PyTorch is to use NCCL - NVIDIA's communication collective library. It doesn't support sparse tensors directly, so we have some work to do.

### Sparse Communication Setting

The outline of the strategy:

Each rank is owns (=responsible for updating) 1/8th of the bigram embedding table, and needs to send to each rank the local gradient rows that have been populated by bigrams it saw, and likewise needs to receive rows it owns from local gradients of other ranks.

Here's a simplified example with two ranks (0,1) and 4 possible bigram hashes [0,1,2,3]

So rank 0 owns bigrams 0,1 and rank 1 owns bigrams 2,3

Rank 0 batch: [1,2]

Rank 1 batch: [1,3]

So rank 0 needs to send to rank 1 its local gradient row 2, and receive row 1.
Likewise, rank 1 needs to send gradient row 1 to rank 0, and receive gradient row 2.

This example has a sparsity ratio of 50%. 

In practice, our 50-batch statistics showed we expect around 80% sparsity, or 20% occupancy.

This aligns pretty well with a measurement: The Value Embeddings parameter happens to be the exact same size as the bigram embeddings, here's how the transfer looks in a profiling session:

![Annotated transfers in profiling](/assets/img/modded-nanogpt/profiling_transfer.png)

The bigram embedding transfer is much closer in duration to the ones for the `lm_head` and `embed` parameters, which happen to be exactly 20% the size of the bigram embedding table, than it is to the value embeddings parameter which is of the same size. This is in the largest batch size - the difference is more marked in earlier in the training run, when batch sizes are smaller.

### Algorithm

Each node needs to receive two things: rows of gradients from other nodes, and their identifiers - which rows they are. 

TODO: pictures!

Our strategy is pretty simple: each node looks at the bigram indexes it saw, and counts how many "belong" to each rank. Then we share those count with all of the other nodes. From this each node can construct a mapping of how many gradient rows it expects to receive from every other node.

Next, we share the actual indexes between all the ranks, using `torch.distributed.all_to_all_single()`.

Finally, we share the actual gradient rows after we have a gradient (=`loss.backward()` finished).

Actually finally, we build a dense gradient from the row-wise sparse gradient we received, which we can then use in our optimizer.

Elided details: what sharing happens on CPU vs GPU and why.

### Future work

#### Technical issues
The above image shows two "holes" which are unrealized gains - places where the GPU is computing but not communicating despite having communication queued. The cause is that GPU communication actually steals a bit of computation, but we are at the mercy of Nvidia's scheduler due to our chosen level of abstraction. By shuffling the transfer initiation order and work order in the optimizer, it should be possible to nudge the scheduler to overlap those computations with tranfers and save about 0.4ms every other step (0.2ms/step, totaling around 0.7 seconds).

A small note here is that a very small portion of the GPU's compute capacity is enough to easily saturate the network links, e.g. Deepseek [reported](https://arxiv.org/pdf/2412.19437v2) (section 3.2.2) allocating a tad under 20% of the GPU for communication ahead of time, in order to maximally overlap computation and communication when training Deepseek V3.

#### More ML-ish

We only optimized half of the communication!

Modded-nanoGPT implements a variant of ZeRO - each rank only optimizes a part of the model, so the structure of cross-gpu communication is:

1. Reduce-scatter phase: share gradients between nodes so each node has the full gradient for its parameters
2. Each rank updates its parameters
3. AllGather: each rank sends its updated parameters to all the others

We implemented a sparse reduce-scatter, but all-gather is still dense, so half of the time-savings are still on the table. Why?

Gather is still dense since our optimizer is dense and updates even weights for bigrams that were not seen: because Adam has a momentum buffer, many weights will have non-zero momenta. Experiments with a sparse optimizer (like [SparseAdam](https://docs.pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html) but with row-level sparsity) showed that the model needs to be re-tuned for this to work. It would save around 0.8ms per Adam step.
