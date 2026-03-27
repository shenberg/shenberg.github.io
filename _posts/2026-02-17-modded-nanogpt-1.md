---
title: "Modded-NanoGPT part 1"
date: 2026-02-17 12:00:00 +0000
slug: modded-nanogpt-1
excerpt: ""
---

I just had the good fortune to slightly improve the [NanoGPT speedrun, modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) record, three separate times. It was harder than I expected.

This will be divided into a few parts:
1. What is NanoGPT, what's a speedrun and why should I care
2. What did I do in order to improve it
3. What did I actually do (in excruciating detail)
4. Wrapping up

<!--more-->

## What is NanoGPT?

Everyone's favorite machine-learning educator, Andrej Karpathy, these days better known for having coined the term vibe-coding, did a series of lecures + exercises called [zero-to-hero](TODO), intended to teach programmers everything necessary to replicate GPT-2, chosen as a model that is still recognizable in today's chatbots. He called his implementation nanoGPT.

### Speedrun?
Karpathy's hook was "$20 to make your own GPT" - Rent a 8xA100 for $20 worth of time and you'll get a model at least equivalent to what OpenAI trained on downstream tasks (caveats: the small configuration, so 124M parameters, not 1.5G, and those $20 exclude the time necessary for setup (downloading the data, tokenizing, etc.)). Those $20 bought about 1.5 hours of 8xA100, and someone on the internet showed that if you run it on 8xH100 you get down to about 45 minutes so probably around $15 at the time, $12 as of now. 

Concretely, he showed that [training on FineWeb](https://github.com/karpathy/llm.c/discussions/481) and reaching a loss of less than 3.28 on FineWeb-val correlates with downstream performance on HellaSwag that's slightly better than the original OpenAI model.

Once that target has been set, the question now becomes, how fast can we do it?
### A good idea?

This is a wall-clock benchmark - how long do you take to reach given model performance on fixed hardware. This is a good measure. The usual method to compare improvements is "We compared our method relative to baseline and trained it for the same number of steps, our results were better." If each step took twice as long, we could have run the baseline for twice the amount of steps, would the new method still have been better?

This is much more pertinent in the LLM training regime, where training is essentially a one-epoch deal (up to resampling of rare types of data), unlike the days of yore when you would train your CNN for X epochs on imagenet, meaning you'd get diminishing returns when going from X to 2X epochs.

Another problem is untuned baselines: Let's choose [n-GPT](https://arxiv.org/abs/2410.01131), which claim basic improvements on transformers by normalizing everything and keeping all activations on the hypersphere. Unfortunately, the baseline is weak (e.g. no QK norm on baseline). I don't mean to pick on them and they're actually good in their methods, for example they tuned the baseline's LR. It's unlikely to be intentional, it's just hard to get a good baseline, unless, there is, say, an entire community is dedicated to making the best, most up-to-date baseline possible.

### A bad idea?

I see two obvious objections to this benchmark:
1. Isn't this just overfitting on the validation set?
2. Why should we care? It won't scale

Both of these are very reasonable objections, and are probably right, to a degree. On the other hand, modded-nanoGPT changes have been ported from the small (124M) to medium (350M) configuration, successfully slashing down training time there too. Later, OG Karpathy, in his newer [nanochat](https://github.com/karpathy/nanochat), ported over many modded-nanogpt changes and got performance better than GPT-2 1.5B in ~3 hours (original training run at OpenAI was 168 hours). Not everything scaled, but enough did.

### How to get started yourself
You need access to an 8 GPU machine, clone [the repository](https://github.com/KellerJordan/modded-nanogpt), paste the 4 lines of shell commands in the readme, hit enter. It should just work.

If, like me, you don't want to spend money on a 8xH100 machine straight away (~$16/hour on [Prime Intellect](https://www.primeintellect.ai/)), it's easiest to rent a smaller amount of H100s (price is roughly linear, expect less than $3/hour for a 1xH100), and modify `run.sh` so that `--nproc_per_node=<Your GPU count>` (it's set at 8).

I worked for the longest time trying ideas on a 1xH100 until I had what seemed like a solid improvement before trying to run it on a full-price machine. You can also wait for times when there are spot instances - they slice prices by about 50%, pretty solid.

## Record #1

Now that I've convinced you it's actually a worthwhile benchmark, we'll get to what I actually did:

1. Changing the multiplication location of the self-attention lambdas to pre-scale the QKV matrix instead of directly scaling V. 
2. Fixing the warm-up process in order to correctly pre-compile all the code paths.

### Multiplication order

As a quick recap, here's some python-ish pseudocode for a GPT:

```python
class GPT:
    # ...
    def forward(self, tokens):
        x = self.embed(tokens[:-1])
        # X shape: batch tokens x Model dimension
        for block in self.blocks:
            x = x + block.self_attn(norm(x)) # Self-Attention
            x = x + block.mlp(norm(x))       # Feed-forward
        logits = self.output(x)
        return cross_entropy(logits, tokens[1:])

class SelfAttention:
    # ...
    def forward(self, x):
        q,k,v = self.Q(x), self.K(x), self.V(x) # linear layers
        # Q,K,V are all of shape batch_tokens x model dim
        # but it may be more appropriate to think of them as shaped as batch_tokens x n_heads x head_dim
        # where head_dim was selected to be model_dim / n_heads
        q = rope(norm(q)) # RoPE, QK-norm
        k = rope(norm(k))
        h = flash_attention(q,k,v) # scaled-dot-product multi-headed attention
        return self.O(h) # output projection
```

One of the architecture modifications performed on GPT-2 is having multiple separate embeddings, called value embeddings, for the input tokens,
that are mixed directly into the self-attention V projections, with learned weights, lambdas.

```python
class SelfAttention:
    # ...
    def forward(self, x, tokens):
        # Same as before

        q,k,v = self.Q(x), self.K(x), self.V(x) # linear layers
        # Q,K,V are all of shape batch_tokens x model dim
        # but it may be more appropriate to think of them as shaped as batch_tokens x n_heads x head_dim
        # where head_dim was selected to be model_dim / n_heads
        q = rope(norm(q)) # RoPE, QK-norm
        k = rope(norm(k))

        # NEW
        ve = self.value_embed(tokens)
        v = v * self.sa_lambda[0] + ve * self.sa_lambda[1]
        # that's it

        h = flash_attention(q,k,v) # scaled-dot-product multi-headed attention
        return self.O(h) # output projection
```

This is a straight-forward `v = v * sa_lambda[0] + ve * sa_lambda[1]`. The value projections (`v` above) are big (32768 by 768 elements on average, depending on batch size). In order to multiply `v` by a scalar, we can intervene in multiple places where we'd need to do less work. The $ W_V $ weight matrix is only 768 by 768, so by changing $ \lambda (W_V x) $ to $ (\lambda W_V) x $ we can save work.

It's still not straight-forward that this would save time as we are running under `torch.compile()`, which could, in theory, fuse the scalar multiply into the $ W_V x $ matrix-multiply kernel by itself. This would hide the cost of the scalar multiplication with the memory accesses for the matrix-multiply. 

This doesn't happen, though, probably because for efficiency reasons*, we keep a single matrix with QKVO and do the QKV projections as a single matrix-multiply.
This does mean we actually pre-multiply the entire QKV matrix by our scalar, which loses efficiency but still works because
the Q and K projections are immediately normed (thanks, QK-norm!).

* efficiency reasons are two: we do the Q,K,V projections in one matrix multiply, and the QKVO matrix is exactly the same size as the MLP
weight matrices, which helps scheduling cross-GPU data transfer as we can mix and match between them and not waste bandwidth on padding.

**Note**: even in layers without value-embeddings, we scale the value projections, and this is probably beneficial for the model as we RMS-norm the input to the SA block, while the residual stream magnitude increases as we reach deeper layers in the network. This allows scaling the output of the SA block. In trained models, the lambdas generally grow with network depth.

### Warm-up fix

#### What's the warm-up phase?

If you've run the speedrun yourself, the first thing you notice is that it actually takes way longer than the advertised wall-clock time. Some of this is installing dependencies, some is downloading the data, but probably the longest part is the first warmup - all of the setup before we get into the actual training loop. 

A big engineering win in the speedrun is to use `torch.compile()` - the current PyTorch JIT compiler. It's a specializing compiler, meaning that
the code it generates will be specialized to the types and shapes of data passed to the function you use, with guards (=ifs) placed to trigger a recompile if the current invocation does not match previous ones. The compilation is pretty time-consuming but is a one-off process, so before the timed training loop, we force the compiler to compile all code-paths we will take during training with all the matrix sizes we will see in training.

#### What was the bug?

A [previous record](https://blog.underfit.ai/nanogpt-record) which improved the DistAdam compute-communication overlap did not update the `torch.compile()` warmup phase to cover all of its newly-added code-paths, which caused a slow-down in the 2nd iteration of training as a recompile was triggered. This happened because there are two optimizers - Muon for linear layers and Adam for everything else, and Adam is only stepped every other training step, accumulating gradients if skipped. This statefulness forced the Adam improvement to add more state, which wasn't correctly set in the warm-up loop, only in the main training loop.

## The path of suffering begins, or, How did I find the improvements?

In short, through lots of pain and suffering (aka learning). My experiment log had about 60 lines before I tried to break a record. Here's a summary of the broad strokes:

When approaching a project like this, I'll usually yolo it and just try stuff, then be systematic.

My first yolo was to use [TREAD](https://github.com/CompVis/tread/), which is a token-masking method which saves a ton of compute in training Diffusion Transformers - it was on my mind from a recent project. The short version is that it didn't work: the increase in speed
was more than made up for in slowing down the convergence of the model. It wasn't very surprising as tokens in images tend not to be as individually important as in language.

The long version is that implementing this was pretty difficult, despite an existing implementation, for a few reasons:
1. The original implementation also shuffled the tokens, which doesn't matter for a DiT as positional embeddings are applied once, before entering the transformer, and the attention is not masked, but we're dealing with causal attention so we're not allowed to reorder tokens.
2. If one generates a random mask on a tensor, one must also appease `torch.compile()` (Dynamo specifically) by telling it that the result of the masking operation will be of a given size, as it cannot infer the resulting tensor sizes by itself. I believe I hit on bugs here too, and I ended up re-writing the algorithm avoiding use of `randperm()` for this.
3. The batch is a 1d sequence of tokens and a sequence of document start/end positions, similar to the old `PackedSequence` of RNNs of the olden days. The start/ends need to be updated according to masking.

After this failure, with the implementation ending up way more time-consuming than expected, I moved on to trying to make incremental changes to the existing architecture of the model.

Stop 1: improving the skip-connection layout for the model, which, at the time, was a U-Net shape with learnable weights. None of my attempts here worked, so I decided to start being systematic.

I trained a model with dense learnable skip connections and then looked at the weights to see if they could teach me something about what would be ideal.

![Dense-connectivity weight visualization](/assets/img/modded-nanogpt/dense_connectivity.png)

The main obvious thing is that layer 8 is special. This is because it's used for another one of the architecture modifications: back-off. It's subtracted from the final layer's output before entering the prediction head - the first 8 layers build context and the last layers predict. It also seems like the first layer has much higher skip weights than the rest.

I modified the skip layout a few ways, never actually improving network performance except if I added more skip connections to layer 8, which ended up roughly break-even between increased runtime and improved convergence. Then I realized the flaw in my plot above - it didn't take into account activation magnitude. I recorded the activations of a trained model and plotted some of them. This made me realize I understood nothing. Here's a plot of activation magnitude per layer (0 is before the 1st layer, 12 is after subtracting the backoff). 

![Activation magnitudes](/assets/img/modded-nanogpt/activation_norms.png)

Activations are HUGE. Looking at the dense connectivity graph, without scaling the skip-connection weight by the ratio of average magnitudes between the source and destination layers, I have no sense of the actual influence of the skip connection on the activation. (Side-note, didn't bother to plot median vs mean as they're pretty much the same, despite the outliers).

By the time I realized this I was already looking at different parts of the network. There was something odd with the dense connectivity matrix: the previous layer's residual connection was effectively down-scaled - the subdiagonal was always slightly negative, meaning each layer, on input, subtracted a bit of the output of the previous layer (=the input). There's already another modification to the model architecture that allows this: in each Transformer layer, the residual `x` is mixed with the original embeddings `x0` as a weighted sum with learned weights, so the residual layer can be down-scaled there already (and is, as has been noted by @classiclary[TODO link]).

The architecture of the model is a pre-norm Transformer, which means we normalize the residual before passing it to self-attention and MLP layers, but the residual itself flows uninterrupted. Now that I knew activation magnitudes were increasing with depth*, I started wondering whether the mixing with x0 was serving a dual purpose of effectively trying to control the residual's magnitude relative to the outputs of the attention and MLP layers (which receive vectors after RMS norm, so magnitude 1). After trying some variations of this structure, I was convinced that both elements of it - residual scaling and mixing with the initial embeddings, are separately beneficial, and also that at least for the self-attention blocks, there is already a scalar for the output - the **value embedding lambdas**. Looking at them closely brought me the improvement.

* This is actually well-known in Transformers and in ResNets too, and can be reasoned about in initialization. With some hand-waving, everything is roughly normally distributed, so we expect the variance of the values in the residual to be some constant times the depth in the network. It was not well-known to me at the time.

## How to break a record, or, the true challenge

By now, I had logged over 50 experiments (I glossed over attempts to improve the MLP layers and other details), where my flow was to use a 1xH100 machine and extrapolate from there to the 8xH100. 

Breaking a record is pretty straight-forward: rent a 8xH100 machine, run your improved version ~10 times, show that final loss is < 3.28 with good statistical significance, time the baseline version on the same machine 3 times to get a relative improvement (unfortunately different machines are not directly comparable despite having the same GPUs with the same interconnect). Not running one command, but not the end of the world. Or so I thought. 

I rented the machine and started recording runs. Almost every run, one or two training steps, usually 60ms, started semi-randomly taking around 400ms longer. It didn't always happen, and the step number varied by 5-10 steps when it did. Unfortunately, reverting to master made it mostly go away (still happened sometimes but very rarely). My code change, which was two lines in a function that's compiled by `torch.compile()`, should really have no such effect, but here we are. After spending a couple of hours gathering data, I realized there's no record run happening - the loss was fine but those extra 400-800ms ate through my efficiency gains. I suspected a hardware issue (you know I'm desperate now), but couldn't find any sort of proof trawling through kernel logs and CUDA debug programs.

This was hard. I had spent a lot of effort, had good reasons to think what I did would work, yet I left with nothing in hand. Quitting seemed reasonable. But before I did, I wrote a github issue detailing my difficulties. happily enough, the core maintainer, @classiclarry[TODO] answered, specifically mentioning that others had encountered the mysterious 400ms step issue in the past.

@classiclarry suspected this was related to some recent distributed optimizer changes, so I learned about that and made some variations on the distributed Adam code, that made sense to me. Unfortunately, not only did they not resolve the problem, they caused the loss to start climbing, even with the code I'd originally tested and had already gotten statistical significance with.

Maybe it's a hardware issue after all and I just wasn't prepared enough to find it in my first attempt? The run is extremely high GPU-utilization, maybe I'm hitting the hosting provider's limits (thermal or power-delivery). I decided to look into NCCL to see if it may have useful logs - maybe only one of the 8 GPU processes was being delayed. There was an apparent slam-dunk running with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL` - a very unexpected log message roughly correlated with the lagging step.

During my investigations of the distributed issue, I had this hypothesis that maybe the ZeRO implementation in distributed Adam was playing a bit too fast-and-loose with parameter ordering and made several changes to the ordering logic, that had no influence either on performance or on the lagging-step issue. **Side-note**: I did discover that, technically, since each GPU has a separate process that does the whole TorchDynamo->TorchInductor compilation process, they are not guaranteed to settle on exactly the same code (The TorchInductor phase can do some profiling and make decisions using the results). On a single machine it's probably fine (there's also a shared disk-backed cache), but perhaps on large-scale runs this is an issue. This is mentioned in [ezyang's State of `torch.compile()` August 2025 post](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/).

I thought up some more logs to check, rented a machine and got perfectly smooth runs. Let's go! Maybe it was just some bad luck in the server lottery after all.

But the code also had this distributed Adam change I didn't love, and I hated that I had no understanding of what went wrong.

I decided to put more effort into figuring out the issue - it being correlated with my code meant maybe some log existed that could help me understand what's going on. I re-checked NCCL, this time from the PyTorch side (`TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL`), and the only thing I could see was that in a lagging step, rank 0 was joining a collective 350ms later than the rest of the ranks, no error.Using pytorch-native logging  it seemed however that it was a wild goose chase, as the lagging step was replicated with no errors in either logs. I believe those logs were caused by not properly shutting down the pytorch process group when interrupting runs in the middle - they were from an older run.

I was getting desperate and running out of ideas, so I went with ChatGPT's suggestion that maybe a recompile was being triggered (this made no sense to me given the nature of the change). One run with `TORCH_LOGS=recompiles` later, this was disproven, but it did discover the second improvement in the record - the warm-up bug.

In the end, I ran out of reasonable ideas for the cause, and ChatGPT's seemed hallucinated (e.g. "rare branch taken" in branchless code). I decided, as a hail-mary, to see whether it's somehow a GC issue. A run with `gc.set_debug(gc.DEBUG_STATS)` showed a memory leak somewhere and one gen-2 collection that does a stop-the-world collection (0.5 sec pause) which roughly fit the timeline.
The problem only disappeared once a `gc.collect()` was added before starting the timer for the run.

I still don't understand why my code change exacerbated the problem in the first place, yet it was now under control. Any ideas as to why a change deep into a compiled function does this are more than welcome. Since `gc.collect()` mitigated the issue, I stopped, roughly as I was digging into the triton kernels generated by `torch.compile()`.

## Next Steps

This was an exercise in humility. In the end I had logged over 60 experiments (not counting debugging runs and some experiments I didn't log) across a wide range of subjects:
* Architecture changes (a mid-2010s DL favorite)
* MLP optimization (interesting ones were "One Wide Feedforward is All You Need" and "Fourier Analysis Networks")
* Activation and gradient control using transformer block variants (Peri-LN, FuseNorm, SandwichNorm)
* Communication-computation overlap improvements
* Low-accuracy training - I was very hopeful for the Collage paper to work out. It did not. For me, that is. You Jiacheng used a float mantissa buffer for bfloat16 training successfully.

Most of the changes were, at best, neutral.

Also, interacting with the community was delightful - people who owe me nothing put in effort both in hypothesizing and in actually testing stuff out in order to help me. The atmosphere was more collaborative than competitive.

It was also humbling - I'm used to feeling, when interacting with my peers, that they're my peers - they're better than me in some stuff, I'm better than them in other stuff, thus we get mutually beneficial exchanges. For the first time in a long time, I felt like an apprentice again, the people I interacted with were just all-around really really good.

Yet, it also showed that sustained efforts work - getting results required mostly believing that I could get results eventually.

This was hard enough that I decided that if I do more, I'll have a better game-plan before starting to work. Turns out this happened twice more.