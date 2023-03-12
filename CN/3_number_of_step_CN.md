Determining the number of steps for each training run

- There are two types of workloads: those that are compute-bound and those that are not.
- When training is compute-bound, training is limited by how long we are willing to wait and not by how much training data we have or some other factor.
	- In this case, if we can somehow train longer or more efficiently, we should see a lower training loss and, with proper tuning, an improved validation loss.
	- In other words, speeding up training is equivalent to improving training and the "optimal" training time is always "as long as we can afford."
	- That said, just because a workload is compute-limited doesn't mean training longer/faster is the only way to improve results.
- When training is not compute-bound, we can afford to train as long as we would like to, and, at some point, training longer doesn't help much (or even causes problematic overfitting).
	- In this case, we should expect to be able to train to very low training loss, to the point where training longer might slightly reduce the training loss, but will not meaningfully reduce the validation loss.
	- Particularly when training is not compute-bound, a more generous training time budget can make tuning easier, especially when tuning learning rate decay schedules, since they have a particularly strong interaction with the training budget.
		- In other words, very stingy training time budgets might require a learning rate decay schedule tuned to perfection in order to achieve a good error rate.
- Regardless of whether a given workload is compute-bound or not, methods that increase the variance of the gradients (across batches) will usually result in slower training progress, and thus may increase the number of training steps required to reach a particular validation loss. High gradient variance can be caused by:
	- Using a smaller batch size
	- Adding data augmentation
	- Adding some types of regularization (e.g. dropout)

确定每个训练运行的步数

- 有两种类型的工作负载：计算密集型和非计算密集型。
- 当训练是计算密集型时，训练受到我们愿意等待的时间限制，而不是训练数据量或其他因素的限制。
	- 在这种情况下，如果我们能够以某种方式更长或更有效地进行训练，我们应该能够看到更低的训练损失和通过适当的调整，改善验证损失。
	- 换句话说，加快训练就相当于改善训练，而“最佳”训练时间始终是“尽可能长”。
	- 尽管如此，仅因为工作负载受计算限制并不意味着延长/加快训练是改善结果的唯一方法。
- 当训练不是计算密集型时，我们可以负担得起训练的时间，但在某些时候，训练时间过长并不能帮助太多（甚至会导致问题性过拟合）。
	- 在这种情况下，我们应该期望能够将训练损失降到非常低的水平，即使延长训练时间可能会略微降低训练损失，但不会显著降低验证损失。
	- 特别是当训练不受计算限制时，更充裕的训练时间预算可以使调整更容易，特别是在调整学习率衰减计划时，因为它们与训练预算有着特别强的相互作用。
		- 换句话说，非常吝啬的训练时间预算可能需要完美调整的学习率衰减计划才能实现良好的错误率。
- 无论给定的工作负载是计算限制还是不受限制，增加梯度的方差（跨批次）的方法通常会导致训练进度较慢，因此可能需要增加训练步骤才能达到特定的验证损失。高梯度方差可能由以下原因引起：
	- 使用较小的批量大小
	- 添加数据增强
	- 添加某些类型的正则化（例如丢弃）

Deciding how long to train when training is not compute-bound
- Our main goal is to ensure we are training long enough for the model to reach the best possible result, while avoiding being overly wasteful in the number of training steps.
- When in doubt, err on the side of training longer. Performance should never degrade when training longer, assuming retrospective (optimal) checkpoint selection is used properly and checkpoints are frequent enough.
- Never tune the max_train_steps number in a study. Pick a value and use it for all trials. From these trials, plot the training step that retrospective checkpoint selection finds in order to refine the choice of max_train_steps.
	- For example, if the best step is always during the first 10% of training, then the maximum number of steps is way too high.
	- Alternatively, if the best step is consistently in the last 25% of training we might benefit from training longer and re-tuning the decay schedule.
- The ideal number of training steps can change when the architecture or data changes (e.g. adding data augmentation).
- Below we describe how to pick an initial candidate value for max_train_steps based on the number of steps necessary to "perfectly fit" the training set using a constant learning rate.
	- Note, we are not using the phrase "perfectly fit the training set" in a precise or mathematically well-defined way. It is merely meant as an informal descriptor to indicate a very low training loss.
		- For example, when training with the log loss, absent regularization terms, we might see the training loss keep slowly improving until we reach floating point limits as the network weights grow without bound and the predictions of the model on the training set become increasingly confident. In this case, we might say the model "perfectly fit" the training set around the time the misclassification error reached zero on the training set.
	- The starting value for max_train_steps we find may need to be increased if the amount of gradient noise in the training procedure increases.
		- For example, if data augmentation or regularizers like dropout are introduced to the model.
	- It may be possible to decrease max_train_steps if the training process improves somehow.
		- For example, with a better tuned optimizer or a better tuned learning rate schedule.

当训练不受计算限制时，如何决定训练时间：

- 我们的主要目标是确保训练足够长，使模型达到最佳结果，同时避免在训练步骤数量上过于浪费。
- 如果有疑虑，应该选择更长的训练时间。假设回顾（最优）检查点选择被正确使用并且检查点足够频繁，训练时间越长，性能不应该下降。
- 在研究中永远不要调整 max_train_steps 数值。选择一个数值并在所有试验中使用它。从这些试验中，可以绘制回顾检查点选择找到的训练步骤，以便优化 max_train_steps 的选择。
	- 例如，如果最佳步骤始终在训练的前10%中，那么最大步数太高了。
	- 或者，如果最佳步骤一直在训练的最后25%中，我们可能会受益于更长的训练时间并重新调整衰减进度。
- 当体系结构或数据发生变化时（例如添加数据增强），理想的训练步数可能会改变。
	- 下面我们将描述如何根据使用恒定学习率“完美拟合”训练集所需的步骤数选择 max_train_steps 的初始候选值。
		- 请注意，我们并没有以精确或数学上定义良好的方式使用“完美拟合训练集”这个短语。它仅仅作为一个非正式的描述符，用于指示训练损失非常低的情况。
			- 例如，在使用对数损失进行训练时，如果没有正则化项，我们可能会看到训练损失保持缓慢改善，直到网络权重无限增长且模型对训练集的预测变得越来越自信，此时我们可能会认为模型在训练集上“完美拟合”。
		- 找到的 max_train_steps 的起始值可能需要增加，如果训练过程中的梯度噪声增加。
			- 例如，如果在模型中引入了数据增强或dropout等正则化器。
		- 如果训练过程有所改善，可能可以降低 max_train_steps。
			- 例如，通过更好地调整优化器或学习率进度。



Algorithm for picking an initial candidate for max_train_steps using a learning rate sweep
[Click to expand]

- This procedure assumes it is possible to not only "perfectly" fit the training set, but to do so using a constant learning rate schedule.
- If it is possible to perfectly fit the entire training set, then there must exist a configuration (with some value of max_train_steps) that perfectly fits the training set; find any such configuration and use its value of max_train_steps as a starting point N.
- Run a constant learning rate sweep (i.e. grid search the learning rate) without data augmentation and without regularization where each trial trains for N steps.
- The number of steps required for the fastest trial in the sweep to reach perfect training performance is our initial guess for max_train_steps.
- NOTE: Bad search spaces can make it possible to engage in self-deception.
	- For example, if all the learning rates in a study are too small, we might incorrectly conclude that a very large value of max_train_steps is necessary.
	- At a minimum, we should check that the optimal learning rate in the study is not at the boundary of the search space.

通过学习率扫描选择最大训练步骤的初始候选算法
[点击展开]

- 该过程假设不仅可以“完美”拟合训练集，而且可以使用恒定的学习率计划来实现。
- 如果能够完全拟合整个训练集，那么必须存在一个配置（具有某个max_train_steps值），可以完美地拟合训练集；找到任何这样的配置并使用其max_train_steps值作为起点N。
- 运行一个常数学习率扫描（即网格搜索学习率），不使用数据增强和正则化，在每个试验中训练N步。
- 在扫描中达到完美的训练表现所需的步骤数，是我们对max_train_steps的初始猜测。
- 注意：糟糕的搜索空间可能会让我们陷入自我欺骗。
	- 例如，如果研究中的所有学习率都太小，则可能会错误地得出需要非常大的max_train_steps值的结论。
	- 至少，我们应该检查研究中的最佳学习率是否在搜索空间的边界上。

Deciding how long to train when training is compute-bound
- In some cases, training loss keeps improving indefinitely and our patience and computational resources become the limiting factors.
- If training loss (or even validation loss) keeps improving indefinitely, should we always train as long as we can afford? Not necessarily.
	- We might be able to tune more effectively by running a larger number of shorter experiments and reserving the longest "production length" runs for the models we hope to launch.
	- As the training time for trials approaches our patience limit, tuning experiments become more relevant for our potential launch candidates, but we can complete fewer of them.
	- There are probably many questions we can answer while only training for ~10% of the production length, but there is always a risk that our conclusions at this time limit will not apply to experiments at 20% of the production length, let alone 100%.
- Tuning in multiple rounds with increasing, per-trial training step limits is a sensible approach.
	- We can do as many rounds as we want, but usually 1-3 are the most practical.
	- Essentially, try to obtain as much understanding of the problem as possible using trials with a very quick turnaround time, trading off tuning thoroughness with relevance to the final, longest runs.
	- Once a given per-trial time limit has generated useful insights, we can increase the training time and continue tuning, double-checking our conclusions from the shorter runs as needed.
- As a starting point, we recommend two rounds of tuning:
	- Round 1: Shorter runs to find good model and optimizer hyperparameters.
	- Round 2: Very few long runs on good hyperparameter points to get the final model.
- The biggest question going from Round i → Round i+1 is how to adjust learning rate decay schedules.
	- One common pitfall when adjusting learning rate schedules between rounds is using all the extra training steps with too small of a learning rate.


- 在某些情况下，训练损失会无限期地改善，而我们的耐心和计算资源成为限制因素。
- 如果训练损失（甚至验证损失）无限期地改善，我们是否应该尽可能地训练？不一定。
	- 通过运行更多的较短实验并将最长的“生产长度”运行保留给我们希望启动的模型，我们可能能够更有效地进行调整。
	- 随着试验的训练时间接近我们的耐心极限，调整实验对我们潜在的启动候选者变得更加相关，但我们可以完成的实验数量较少。
	- 在训练约10%的时间内，我们可能可以回答许多问题，但在此时间限制下得出的结论往往不适用于训练长度为20%或100%的实验。
- 使用逐步增加的每次试验训练步骤限制进行多轮调整是明智的方法。
	- 我们可以进行任意多轮调整，但通常最实用的是1-3轮。
	- 本质上，尝试使用非常快的周转时间进行尝试，以换取调整的彻底性和与最终的最长运行的相关性，尽可能地了解问题。
	- 一旦特定的每次试验时间限制生成有用的见解，我们就可以增加训练时间并继续调整，并根据需要检查较短运行的结论。
- 作为起点，我们建议进行两轮调整：
	- 第一轮：运行较短时间以找到良好的模型和优化器超参数。
	- 第二轮：在良好的超参数点上进行很少的长时间运行以获得最终模型。
- 从第i轮到第i+1轮的最大问题是如何调整学习率衰减计划。
	- 调整学习率计划之间的一个常见陷阱是使用所有额外的训练步骤并使用过小的学习率。

Round 1
[Click to expand]

- Unfortunately, there is no guarantee that good hyperparameters found in short, incomplete training are still good choices when training length is significantly increased. However, for some kinds of hyperparameters, they are often correlated enough for Round 1 to be useful.
- What hyperparameter values found in shorter runs do we expect to transfer to longer training runs? For all of this, we need more research. But based on what we know so far, here are the authors’ suspicions in order of decreasing probability of transferring:
	- Very likely to transfer
		- Early training instability can be resolved in the first round of tuning using a smaller number of training steps. Perhaps these hyperparameters are the closest thing to a sure bet for transfer that we have.
			- Warmup length
			- Initialization
	- Likely to transfer
		- Model architecture - A dramatic win in the model architecture will usually transfer, but there are probably many counterexamples.
	- Might transfer
		- Optimization algorithm/optimizer hyperparameters - We think this would "loosely" transfer. It’s definitely weaker than the things above it.
		- Data augmentation
		- Regularization
			- If it isn't possible to perfectly fit the training set, the model might be in a regime where regularization is unlikely to help very much.
	- Unlikely to transfer
		- Learning rate schedule: unlikely to transfer perfectly.
			- This paper suggests that even decay schedule transfers, but we don't believe this is true in general. Example: Tuning sqrt decay on small # of training steps then extending to large # will result in the majority of training occurring at overly small steps.
				- One can likely do "good enough" with most schedules in the limit of extreme training budget, but noticeable performance improvements can likely be seen if it is tuned.
		- Understanding Short-Horizon Bias in Stochastic Meta-Optimization describes the dangers of trying to pick learning rates myopically.

第一轮

[点击展开]

- 不幸的是，在短时间内进行的不完整的训练中找到的良好超参数，在训练时长显著增加时并不能保证仍然是良好的选择。然而，对于某些超参数类型来说，它们通常足够相关，以至于第一轮调整仍然是有用的。
- 我们期望将哪些在短时间运行中找到的超参数值转移到更长的训练运行中？关于这一切，我们需要更多的研究。但是根据我们目前所知，这里是作者们怀疑的转移概率递减的顺序：
	- 很可能转移
		- 在第一轮调整中，可以使用较少的训练步骤来解决早期训练不稳定的问题。也许这些超参数是我们拥有的最接近确定的转移选择。
			- 热身长度
			- 初始化
	- 可能转移
		- 模型结构 - 模型结构的显著改进通常会转移，但可能存在许多反例。
	- 可能转移
		- 优化算法/优化器超参数 - 我们认为这可能会“松散”转移。它肯定比上述内容要弱。
		- 数据增强
		- 正则化
			- 如果无法完美拟合训练集，模型可能处于正则化很难起作用的区域。
	- 不太可能转移
		- 学习率调度：不太可能完美转移。
			- 这篇论文表明即使衰减调度转移，但我们不相信这在一般情况下是正确的。例如：在少量的训练步骤上调整sqrt衰减，然后将其扩展到大量的步骤，将导致大部分训练以过小的步骤进行。
				- 在极端的训练预算限制下，大多数调度可能都足够好，但如果进行调整，可以明显地看到性能的改善。
		- 在随机元优化中理解短视偏差描述了试图短视地选择学习率的危险。

Round 2
[Click to expand]

- Run the best hyperparameter configuration from Round 1.
- (Speculation) 🤖 Use the extra steps to extend the period of training at a high learning rate.
	- E.g. if linear schedule then keep the length of the decay fixed from Round 1 and extend the period of constant lr in the 	beginning.
	- For cosine decay, just keep the base lr from Round 1 and extend max_train_steps as in Chinchilla paper.
- More rounds might make sense for teams with very mature modeling and tuning pipelines and very long and expensive production training runs, but they will often be overkill.
	- We've described how to transfer from Step 1 → Step 2. If we didn't care about analysis time and if making efficient use of compute was the overriding concern, then the ideal would be to exponentially increase the length of training runs (and thus the end-to-end time to complete a study) over many different rounds of tuning.
		- At each round we systematically ensure our choices continue to hold up.
		- New ideas go through a pipeline that progressively derisks them using increasingly long-running experiments from Step i to Step i+1.

第二轮
[点击展开]

- 运行第一轮的最佳超参数配置。
- （推测）🤖 使用额外的步骤延长以高学习率训练的时间。
	- 例如，如果使用线性计划表，则保持从第一轮开始的衰减长度不变，并延长开始时的恒定学习率期间。
	- 对于余弦衰减，只需保持第一轮的基础学习率，并将最大训练步骤延长，如Chinchilla paper中所述。
- 对于具有非常成熟的建模和调优流水线以及非常长和昂贵的生产训练运行的团队来说，更多轮次可能是有意义的，但它们经常是多余的。
	- 我们已经描述了如何从步骤1 → 步骤2进行转移。如果我们不关心分析时间，并且利用计算机的效率是主要关注点，那么理想情况是在许多不同的调优轮次中指数增加训练时间（因此是完成研究的端到端时间）。
		- 在每一轮中，我们系统地确保我们的选择仍然可靠。
		- 新的想法通过逐渐增加的从步骤i到步骤i+1的长时间实验流程进行风险分散。

FAQs
What is the best learning rate decay schedule family?
[Click to expand]

- It’s an open problem. It’s not clear how to construct a set of rigorous experiments to confidently answer what the "best" LR decay schedule is.
- Although we don't know the best schedule family, we're confident that it’s important to have some (non-constant) schedule and that tuning it matters.
- Different learning rates work best at different times during the optimization process. Having some sort of schedule makes it more likely for the model to hit a good learning rate.

什么是最好的学习率衰减时间表族？

- 这是一个开放性的问题。目前尚不清楚如何构建一组严格的实验来自信地回答什么是“最好的”学习率衰减时间表。
- 虽然我们不知道最佳的时间表族，但我们有信心需要一些（非常量）时间表，并且调整它很重要。
- 不同的学习率在优化过程的不同阶段表现最佳。拥有某种时间表使模型更有可能达到良好的学习率。

Which learning rate decay should I use as a default?
[Click to expand]

Our preference is either linear decay or cosine decay, and a bunch of other schedule families are probably good too.

我应该将哪种学习率衰减作为默认值？

- 我们的首选是线性衰减或余弦衰减，其他一些调度族可能也很好

Why do some papers have complicated learning rate schedules?
[Click to expand]

- It’s not uncommon to see papers with complicated piecewise learning rate (LR) decay schedules.
- Readers often wonder how the authors arrived at such a complicated study.
- Many complicated LR decay schedules are the result of tuning the schedule as a function of the validation set performance in an ad hoc way:
	- Start a single training run with some simple LR decay (or a constant learning rate).
	- Keep training running until the performance seems to stagnate. If this happens, pause training. Resume it with a perhaps steeper LR decay schedule (or smaller constant learning rate) from this point. Repeat this process until the conference/launch deadline.
- Blithely copying the resulting schedule is generally not a good idea since the best particular schedule will be sensitive to a host of other hyperparameter choices.
	- Better to copy the algorithm that produced the schedule, although this is rarely possible when arbitrary human judgment produced the schedule.
- This type of validation-error-sensitive schedule is fine to use if it can be fully automated, but human-in-the-loop schedules that are a function of validation error are brittle and not easily reproducible, so we recommend avoiding them.
	- Before publishing results that used such a schedule, please try to make it fully reproducible.


为什么一些论文中有复杂的学习率调度？

- 论文中出现复杂的分段学习率(LR)衰减调度并不罕见。
- 读者常常会想知道作者是如何得出这样复杂的研究结果的。
- 许多复杂的LR衰减调度是通过将调度作为验证集性能的函数进行调整而得到的：
	- 开始一个简单的LR衰减（或恒定的学习率）的单次训练运行。
	- 进行训练，直到性能似乎停滞。如果发生这种情况，请暂停训练。从此时开始，使用更陡峭的LR衰减调度（或更小的恒定学习率）恢复训练。重复此过程，直到会议/发布截止日期。
- 盲目地复制结果调度通常不是一个好主意，因为最佳的特定调度会对一系列其他超参数选择敏感。
	- 最好复制生成调度的算法，但当任意的人类判断产生调度时，这通常是不可能的。
- 如果可以完全自动化使用这种验证误差敏感调度，那么使用它是可以接受的，但是那些作为验证误差函数的人为调度是脆弱的，不易重现，因此我们建议避免使用它们。
	- 在发布使用这种调度的结果之前，请尝试使其完全可重现。


How should Adam’s hyperparameters be tuned?
[Click to expand]

As discussed above, making general statements about search spaces and how many points one should sample from the search space is very difficult. Note that not all the hyperparameters in Adam are equally important. The following rules of thumb correspond to different "budgets" for the number of trials in a study.
- If < 10 trials in a study, only tune the (base) learning rate.
- If 10-25 trials, tune learning rate and .
If 25+ trials, tune the learning rate, 
 and 
.
If one can run substantially more than 25 trials, additionally tune 
.


如何调整Adam的超参数？

- 如上所述，对于搜索空间以及应从搜索空间采样多少点的一般性陈述非常困难。请注意，Adam中并非所有超参数的重要性相同。以下经验法则对应于研究中试验数量的不同“预算”：

	- 如果研究中的试验次数小于10次，则仅调整（基本）学习率。

	- 如果有10-25次试验，则调整学习率和β1。

	- 如果有25次以上的试验，则调整学习率、β1和β2。

	- 如果可以运行远远超过25次的试验，则还需调整ε。



Why use quasi-random search instead of more sophisticated black box optimization algorithms during the exploration phase of tuning?
[Click to expand]
- Quasi-random search (based on low-discrepancy sequences) is our preference over fancier black box optimization tools when used as part of an iterative tuning process intended to maximize insight into the tuning problem (what we refer to as the "exploration phase"). Bayesian optimization and similar tools are more appropriate for the exploitation phase.
- Quasi-random search based on randomly shifted low-discrepancy sequences can be thought of as "jittered, shuffled grid search", since it uniformly, but randomly, explores a given search space and spreads out the search points more than random search.
- The advantages of quasi-random search over more sophisticated black box optimization tools (e.g. Bayesian optimization, evolutionary algorithms) include:
	- Sampling the search space non-adaptively makes it possible to change the tuning objective in post hoc analysis without rerunning experiments.
		- For example, we usually want to find the best trial in terms of validation error achieved at any point in training. But the non-adaptive nature of quasi-random search makes it possible to find the best trial based on final validation error, training error, or some alternative evaluation metric without rerunning any experiments.
	- Quasi-random search behaves in a consistent and statistically reproducible way.
		- It should be possible to reproduce a study from six months ago even if the implementation of the search algorithm changes, as long as it maintains the same uniformity properties. If using sophisticated Bayesian optimization software, the implementation might change in an important way between versions, making it much harder to reproduce an old search. It isn’t always possible to roll back to an old implementation (e.g. if the optimization tool is run as a service).
Its uniform exploration of the search space makes it easier to reason about the results and what they might suggest about the search space.
For example, if the best point in the traversal of quasi-random search is at the boundary of the search space, this is a good (but not foolproof) signal that the search space bounds should be changed. This section goes into more depth. However, an adaptive black box optimization algorithm might have neglected the middle of the search space because of some unlucky early trials even if it happens to contain equally good points, since it is this exact sort of non-uniformity that a good optimization algorithm needs to employ to speed up the search.
Running different numbers of trials in parallel versus sequentially will not produce statistically different results when using quasi-random search (or other non-adaptive search algorithms), unlike with adaptive algorithms.
More sophisticated search algorithms may not always handle infeasible points correctly, especially if they aren't designed with neural network hyperparameter tuning in mind.
Quasi-random search is simple and works especially well when many tuning trials will be running in parallel.
Anecdotally1, it is very hard for an adaptive algorithm to beat a quasi-random search that has 2X its budget, especially when many trials need to be run in parallel (and thus there are very few chances to make use of previous trial results when launching new trials).
Without expertise in Bayesian optimization and other advanced black box optimization methods, we might not achieve the benefits they are, in principle, capable of providing. It is hard to benchmark advanced black box optimization algorithms in realistic deep learning tuning conditions. They are a very active area of current research, and the more sophisticated algorithms come with their own pitfalls for inexperienced users. Experts in these methods are able to get good results, but in high-parallelism conditions the search space and budget tend to matter a lot more.
That said, if our computational resources only allow a small number of trials to run in parallel and we can afford to run many trials in sequence, Bayesian optimization becomes much more attractive despite making our tuning results harder to interpret.
为什么在调参的探索阶段使用准随机搜索而不是更复杂的黑箱优化算法？

- 在我们的迭代调优过程中，准随机搜索（基于低失真序列）是我们比喜欢的黑箱优化工具更适合于旨在最大化对调优问题的洞察力的探索阶段。贝叶斯优化和类似的工具更适合于利用阶段。

- 基于随机移位的低失真序列的准随机搜索可以被认为是“抖动的、洗牌的网格搜索”，因为它均匀但随机地探索给定的搜索空间，并比随机搜索更广泛地分布搜索点。


- 准随机搜索相比于更复杂的黑盒优化工具（例如贝叶斯优化、进化算法）的优点包括：
	- 非自适应地对搜索空间进行抽样，使得在事后分析中可以更改调整目标而无需重新运行实验。
		- 例如，我们通常希望在训练的任何时候找到在验证误差方面取得的最佳试验。但是，准随机搜索的非自适应性使得在不重新运行任何实验的情况下，可以基于最终验证误差、训练误差或某些其他评估指标找到最佳试验。
	- 准随机搜索以一致和统计可重复的方式行为。
		-如果保持相同的均匀性属性，应该能够重现六个月前的研究，即使搜索算法的实现发生了变化。如果使用复杂的贝叶斯优化软件，实现在版本之间可能会发生重大变化，这使得难以重现旧的搜索。有时无法回滚到旧的实现（例如，如果优化工具作为服务运行）。
	- 其对搜索空间的均匀探索使得更容易推断结果及其对搜索空间的建议。
		- 例如，如果准随机搜索遍历中的最佳点位于搜索空间的边界，则这是一个好的（但不是绝对准确的）信号，表明应该更改搜索空间的边界。本节将深入探讨此问题。然而，一个自适应的黑盒优化算法可能会因为某些不幸的早期试验而忽略了搜索空间的中间部分，即使它包含同样好的点，因为正是这种不均匀性使得良好的优化算法需要使用来加速搜索。

- 使用准随机搜索（或其他非自适应搜索算法）并行运行不同数量的试验与按顺序运行不会产生统计上的不同结果，这与自适应算法不同。
- 更复杂的搜索算法可能无法正确处理不可行点，特别是如果它们没有考虑神经网络超参数调整而设计。

- 准随机搜索简单易行，特别适用于许多调整试验将并行运行的情况。
	- 据说，在很多试验需要并行运行（因此在启动新试验时很少有机会利用以前的试验结果）的情况下，很难让自适应算法击败其预算2倍的准随机搜索。
	- 如果没有贝叶斯优化和其他高级黑盒优化方法的专业知识，我们可能无法实现它们原则上能够提供的好处。在实际深度学习调整条件下，很难对高级黑盒优化算法进行基准测试。它们是当前研究的一个非常活跃的领域，而更复杂的算法对于没有经验的用户来说也有其自身的缺陷。这些方法的专家能够获得良好的结果，但在高并行条件下，搜索空间和预算往往更为重要。

- 话虽如此，如果我们的计算资源只允许并行运行少量试验，并且我们可以负担运行许多试验序列，那么贝叶斯优化将变得更加有吸引力，尽管会使我们的调整结果更难以解释。



Where can I find an implementation of quasi-random search?
[Click to expand]

- Open-Source Vizier has an implementation of quasi-ranom search. Set algorithm="QUASI_RANDOM_SEARCH" in this usage example.
- An alternative implementation exists here.
- Both implementations above generate a Halton sequence for a given search space (intended to implement a shifted, scrambled Halton sequence as recommended in https://arxiv.org/abs/1706.03200).
- If a quasi-random search algorithm based on a low-discrepancy sequence is not available, it is possible to substitute pseudo random uniform search instead, although this is likely to be slightly less efficient.
	- In 1-2 dimensions, grid search is also acceptable, although not in higher dimensions (see Bergstra & Bengio, 2012).

我在哪里可以找到准随机搜索的实现？

- 开源的Vizier中有一个准随机搜索的实现。在此使用示例中设置algorithm="QUASI_RANDOM_SEARCH"。
- 另外一个实现在此处。
- 上述两个实现都为给定的搜索空间生成Halton序列（旨在实现https://arxiv.org/abs/1706.03200中推荐的偏移、混淆的Halton序列）。
- 如果基于低差异序列的准随机搜索算法不可用，则可以替换为伪随机均匀搜索，尽管这可能略微低效。
	- 在1-2维中，网格搜索也是可接受的，但在高维中不可行（参见Bergstra和Bengio，2012）。



How many trials are needed to get good results with quasi-random search?
[Click to expand]

A box plot showing the importance of sampling enough

Figure 3: A ResNet-50 was tuned on ImageNet with 100 trials. Via bootstrapping, different amounts of tuning budget were simulated. Box plots of the best performances for each trial budget are plotted above.

There is no way to answer this question in general, but we can look at specific examples.
As the Figure 3 shows, the number of trials in a study can have a substantial impact on the results.
Notice how large the interquartile ranges are when 6 trials were sampled, versus when 20 trials were sampled.
Even with 20 trials, it is likely that the difference between especially lucky and unlucky studies will be larger than the typical variation between re-trains of this model on different random seeds, with fixed hyperparameters, which for this workload might be around +/- 0.1% on a validation error rate of ~23%.

使用准随机搜索需要多少次试验才能获得好结果？

- 无法一般性地回答这个问题，但我们可以看具体的例子。
- 如图3所示，研究中的试验次数可以对结果产生重大影响。
	- 请注意，当采样6次试验时，四分位距是多么大，而当采样20次试验时则不同。
	- 即使采样了20次试验，特别是幸运或不幸的研究之间的差异可能会比该工作负载上该模型在不同随机种子、固定超参数下重新训练的典型差异更大，后者的验证误差率大约为23%左右，误差率变化范围可能在+/- 0.1%左右。



How can optimization failures be debugged and mitigated?
[Click to expand]

- Summary: If the model is experiencing optimization difficulties, it’s important to fix them before trying other things. Diagnosing and correcting training failures is an active area of research.

Changing the strides in a single residual block in a WideResnet results in training instability.

- Figure 4: Changing the strides in a single residual block (2x2 -> 1x1) in a WideResnet results in training instability. This does not degrade performance at low learning rates, but high learning rates no longer train well due to the instability. Applying 1000 steps of learning rate warmup resolves this particular instance of instability, allowing stable training at max learning rate of .1.
如何调试和缓解优化失败问题？

- 总结：如果模型遇到优化困难，重要的是在尝试其他方法之前解决这些困难。诊断和纠正训练失败是研究的一个活跃领域。
在WideResnet中更改单个残差块的步幅会导致训练不稳定。

- 图4：在WideResnet中更改单个残差块（2x2-> 1x1）的步幅会导致训练不稳定。这不会降低低学习率下的性能，但由于不稳定性，高学习率不再训练良好。应用1000步学习率预热可以解决这种特定的不稳定性实例，从而允许在最大学习率为0.1时稳定训练。

- Identifying unstable workloads
- Any workload will become unstable if the learning rate is too large. Instability is only an issue when it forces us to use a learning rate that’s too small.
- There are at least two types of training instability worth distinguishing:
	- Instability at initialization/early in training.
	- Sudden instability in the middle of training.
- We can take a systematic approach to identifying stability issues in our workload.
	- Do a learning rate sweep and find the best learning rate lr*.
	- Plot training loss curves for learning rates just above lr*.
	- If the learning rates > lr* show loss instability (loss goes up not down during periods of training), then it is likely that fixing the instability will result in better training.
- Log the L2 norm of the full loss gradient during training, outlier values can result in spurious instability in the middle of training. This can inform how to pick gradient/update clipping.

NOTE: Some models show very early instability followed by a recovery that results in slow but stable training. Common evaluation schedules can miss these issues by not evaluating frequently enough!
- 识别不稳定工作负载
- 如果学习率过大，任何工作负载都会变得不稳定。只有当不稳定性迫使我们使用太小的学习率时才会成为问题。
- 至少有两种值得区分的训练不稳定类型：
	- 初始化/训练早期的不稳定性。
	- 训练中期突然出现的不稳定性。
- 我们可以采用系统性方法来识别工作负载中的稳定性问题。
	- 进行学习率扫描，找到最佳学习率lr*。
	- 绘制学习率略高于lr*的训练损失曲线。
	- 如果学习率> lr*显示损失不稳定性（在训练期间损失上升而不是下降），则修复不稳定性可能会导致更好的训练。
- 在训练期间记录完整损失梯度的L2范数，异常值可能会导致训练中出现虚假的不稳定性。这可以指导如何选择梯度/更新剪裁。
注意：有些模型显示出早期不稳定性，随后恢复正常，导致训练缓慢但稳定。常见的评估计划可能由于评估频率不够而忽略这些问题！

To check for this, we can train for an abbreviated run of just ~500 steps using lr = 2 * current best, but evaluate every step.
Figure 5: Illustration of the value of more frequent evaluations at the start of training. Useful if there’s a suspicion that the model suffers from early training instability.

为了检查这一点，我们可以使用lr = 2 *当前最佳值进行缩短的约500个步骤的训练，但每步都进行评估。
图5：更频繁地在训练开始时进行评估的价值说明。如果怀疑模型在训练初期存在不稳定性，则这是有用的。

- Potential fixes for common instability patterns
	- Apply learning rate warmup
		- Best for early training instability.
	- Apply gradient clipping
		- Good for both early and mid training instability, may fix some bad inits that warmup cannot.
	- Try a new optimizer
		- Sometimes Adam can handle instabilities that Momentum can’t. This is an active area of research.
	- We can ensure that we’re using best practices/initializations for our model architecture (examples below).
		- Add residual connections and normalization if the model doesn't contain it already.
	- Normalization should be the last operation before the residual. E.g. x + Norm(f(x)).
	- Norm(x + f(x)) known to cause issues.
	- Try initializing residual branches to 0 (e.g. ReZero init).
	- Lower the learning rate
		- This is a last resort.
- Learning rate warmup

- 常见不稳定性模式的潜在修复方案
	- 应用学习率预热
		- 最适合早期训练的不稳定性。
	- 应用梯度裁剪
		- 既适合早期训练的不稳定性，也适合中期训练的不稳定性，可能修复一些预热无法解决的问题。
	- 尝试新的优化器
		- 有时Adam能够处理Momentum无法处理的不稳定性。这是一个活跃的研究领域。
	- 我们可以确保我们在使用最佳实践/初始化来实现我们的模型架构（示例如下）。
		- 如果模型中没有，添加残差连接和归一化。
	- 归一化应该是在残差之前的最后一个操作。例如x + Norm(f(x))。
	- Norm(x + f(x))被认为会导致问题。
	- 尝试将残差分支初始化为0（例如ReZero初始化）。
	- 降低学习率
		- 这是最后的手段。
- 学习率预热

Figure 6: An example of instability during a warmup period (note the horizontal axis log scale). 40k steps of warmup was needed for successful training in this case.

图6：热身期间不稳定性的示例（请注意水平轴对数刻度）。在这种情况下，需要进行40k步的热身才能成功训练。

- When to apply learning rate warmup

Figure 7a: An example of a hyperparameter axis plot for a model exhibiting training instability. The best learning rate is at the edge of what is feasible. An "infeasible" trial is defined as one that either produces NaNs or uncharacteristically high values of the loss.
	- Figure 7a shows a hyperparameter axis plot that indicates a model experiencing optimization instabilities, because the best learning rate is right at the edge of instability.
	- Figure 7b shows how this can be double-checked by examining the training loss of a model trained with a learning rate either 5x or 10x larger than this peak. If that plot shows a sudden rise in the loss after a steady decline (e.g. at step ~10k in the figure above), then the model likely suffers from optimization instability.

- 何时应用学习率热身
图7a：展示了一个模型训练不稳定性的超参数轴图示。最佳学习率位于可行范围的边缘。在该图中，“不可行的”试验定义为产生NaN或异常高损失值的试验。

- 图7a显示了一个超参数轴图示，指示了一个模型经历了优化不稳定性，因为最佳学习率恰好处于不稳定性的边缘。
- 图7b展示了如何通过检查使用比该峰值大5倍或10倍的学习率训练的模型的训练损失来进行双重检查。如果该图显示在稳定下降后突然出现损失的上升（例如在上面的图中的步骤~10k处），则该模型可能遭受优化不稳定性的困扰。

- How to apply learning rate warmup
	- Using the section immediately above, we assume that the practitioner has already identified the learning rate at which the model becomes unstable. This is the unstable_base_learning_rate.
	- Warmup involves prepending a learning rate schedule that ramps up the learning rate from 0 to some stable base_learning_rate, that is at least one order of magnitude larger than unstable_base_learning_rate. The default would be to try a base_learning_rate that’s 10x unstable_base_learning_rate. Although note that it’d be possible to run this entire procedure again for something like 100x unstable_base_learning_rate. The specific schedule is:
		- Ramp up from 0 to base_learning_rate over warmup_steps.
		- Train at a constant rate for post_warmup_steps.
	- Our goal is to find the shortest number of warmup_steps that allows us to access peak learning rates that are much higher than unstable_base_learning_rate.
	- So for each base_learning_rate, we need to tune warmup_steps and post_warmup_steps. It’s usually fine to set post_warmup_steps to be 2*warmup_steps.
	- Warmup can be tuned independently of an existing decay schedule. warmup_steps should be swept at a few different orders of magnitude. For example, an example study could try [10, 103, 104, 105]. The largest feasible point shouldn't be more than 10% of max_train_steps.
	- Once a warmup_steps that doesn't blow up training at base_learning_rate has been established, it should be applied to the baseline model. Essentially, we prepend this schedule onto the existing schedule, and use the optimal checkpoint selection discussed above to compare this experiment to the baseline. For example, if we originally had 10,000 max_train_steps and did warmup_steps for 1000 steps, the new training procedure should run for 11,000 steps total.
	- If long warmup_steps are required for stable training (>5% of max_train_steps), max_train_steps may need to be increased to account for this.
	- There isn't really a "typical" value across the full range of workloads. Some models only need 100 steps, while others (particularly transformers) may need 40k+.

- 如何应用学习率预热
	- 假设从上面所述，实践者已经确定了模型变得不稳定的学习率。这个不稳定的基础学习率就是 unstable_base_learning_rate。
	- 预热涉及准备一个学习率调度表，将学习率从0逐步升高到一些稳定的基础学习率，至少比 unstable_base_learning_rate 高一阶。默认值为尝试10倍于 unstable_base_learning_rate 的基础学习率。但请注意，可以针对像 100 倍 unstable_base_learning_rate 这样的倍数再运行整个过程。具体的时间表是：
		- 在 warmup_steps 中将学习率从0逐渐增加到 base_learning_rate。
		- 在 post_warmup_steps 中以恒定速率训练。
	- 我们的目标是找到最短的 warmup_steps，以便能够访问远高于 unstable_base_learning_rate 的最大学习率。
	- 因此，对于每个基础学习率，我们需要调整 warmup_steps 和 post_warmup_steps。通常将 post_warmup_steps 设置为 2 * warmup_steps 即可。
	- 预热可以独立于现有衰减时间表进行调整。应在几个不同数量级下扫描 warmup_steps。例如，一个示例研究可以尝试 [10、103、104、105]。最大可行点不应超过 max_train_steps 的 10%。
	- 一旦确定了一个 warmup_steps，在 base_learning_rate 上进行训练不会导致训练崩溃，就应将其应用于基线模型。实质上，我们将此时间表添加到现有时间表之前，并使用上面讨论的最优检查点选择来将此实验与基线进行比较。例如，如果我们最初有 10,000 个 max_train_steps，并进行了 1000 个步骤的 warmup_steps，则新的训练过程应该总共运行 11,000 个步骤。
	- 如果需要长时间的 warmup_steps 进行稳定训练（> max_train_steps 的 5%），则可能需要增加 max_train_steps 来解决这个问题。
	- 在整个工作负载的全范围内并没有真正的“典型”值。有些模型只需要 100 步，而其他模型（特别是 transformers）可能需要 40k+ 步。


- Gradient clipping
- Gradient clipping is most useful when large or outlier gradient issues occur.
- Clipping can fix either early training instability (large gradient norm early), or mid training instabilities (sudden gradient spikes mid training).
- Sometimes longer warmup periods can correct instabilities that clipping does not: see this section above.
	🤖 What about clipping during warmup?
- The ideal clip thresholds are just above the "typical" gradient norm.
- Here’s an example of how gradient clipping could be done:
	- If the norm of the gradient ｜g｜is greater than the gradient clipping threshold lambda, then do 
 
 where 
 is the new gradient.
- Log the unclipped gradient norm during training. By default, generate:
	- A plot of gradient norm vs step
	- A histogram of gradient norms aggregated over all steps
- Choose a gradient clipping threshold based on the 90th percentile of gradient norms.
	- The threshold will be workload dependent, but 90% is a good starting point. If it doesn't work, this threshold can be tuned.
	- 🤖 What about some sort of adaptive strategy?
- If we try gradient clipping and the instability issues remain, we can try it harder (i.e. make the threshold smaller).
- Extremely aggressive gradient clipping is in essence a strange way of reducing the learning rate. If we find ourselves using extremely aggressive clipping, we probably should just cut the learning rate instead.
- We would usually consider having >50% of the updates getting clipped somehow as "extremely aggressive".
- If we need to do extremely aggressive gradient clipping to deal with our instability issues, then we might as well reduce the learning rate.

- Gradient clipping（梯度裁剪）
	- 当梯度出现异常大或离群值时，梯度裁剪最有用。
	- 裁剪可以解决早期训练不稳定性（早期大梯度范数）或中期训练不稳定性（中期突然梯度波动）。
	- 有时较长的预热期可以纠正裁剪无法纠正的不稳定性：请参见上面的这节。
	- 🤖预热期间的裁剪怎么样？
	- 理想的裁剪阈值略高于“典型”梯度范数。
	- 这是梯度裁剪的一个例子：
		- 如果梯度范数 ｜g｜大于梯度裁剪阈值 lambda，则进行以下操作g^{\prime}=\lambda \times \frac{g}{|g|}
		其中g^{\prime}
		是新梯度。

	- 在训练期间记录未裁剪的梯度范数。默认生成：
		- 梯度范数 vs 步数 的绘图
		- 所有步骤的梯度范数的直方图
	- 根据梯度范数的第90个百分位数选择梯度裁剪阈值。
		- 阈值将取决于工作负载，但90％是一个好的起点。如果不起作用，可以调整此阈值。
		- 🤖使用某种自适应策略怎么样？
	- 如果尝试梯度裁剪而不稳定性问题仍然存在，我们可以尝试更严格的裁剪（即减小阈值）。
	- 极度激进的梯度裁剪本质上是一种减少学习率的奇怪方式。如果我们发现自己正在使用极度激进的裁剪，则可能应该直接降低学习率。
	- 我们通常认为，将>50％的更新以某种方式进行裁剪称为“极度激进”。
	- 如果我们需要进行极度激进的梯度裁剪来处理不稳定性问题，那么我们可以考虑降低学习率。

Why do you call the learning rate and other optimization parameters hyperparameters? They are not parameters of any prior distribution.
[Click to expand]

- It is true that the term "hyperparameter" has a precise meaning in Bayesian machine learning and referring to the learning rate and most of the other parameters we tune in deep learning as "hyperparameters" is an abuse of terminology.
- We would prefer to use the term "metaparameter" for learning rates, architectural parameters, and all the other things we tune in deep learning, since it avoids the potential for confusion that comes from misusing the word "hyperparameter" (confusion that is especially likely when discussing Bayesian optimization where the probabilistic response surface models have their own true hyperparameters).
- Unfortunately, although potentially confusing, the term hyperparameter has become extremely common in the deep learning community.
- Therefore, for a document, such as this one, intended for a wide audience that includes many people who are unlikely to be aware of this technicality, we made the choice to contribute to one source of confusion in the field in hopes of avoiding another.
That said, we might make a different choice when publishing a research paper, and we would encourage others to use "metaparameter" instead in most contexts.


为什么你把学习率和其他优化参数称为超参数？它们不是任何先前分布的参数。

- 确实，在贝叶斯机器学习中，“超参数”这个术语有一个精确的含义，将学习率和大多数其他我们在深度学习中调整的参数称为“超参数”是术语的滥用。
- 我们更倾向于使用“元参数”这个术语来描述学习率、架构参数和所有其他在深度学习中调整的参数，因为它避免了错误使用“超参数”这个词可能带来的混淆（特别是在讨论贝叶斯优化时，概率响应面模型有自己的真实超参数）。
- 不幸的是，尽管可能会引起混淆，但“超参数”这个术语在深度学习社区中已经非常普遍。
- 因此，对于像这样旨在面向包括许多不太可能了解这种技术细节的人群的文档，我们选择在领域中为一个来源的混淆做出贡献，以避免另一个混淆。
尽管如此，当发布研究论文时，我们可能会做出不同的选择，并鼓励其他人在大多数情况下使用“元参数”代替“超参数”。


Why shouldn't the batch size be tuned to directly improve validation set performance?
[Click to expand]

- Changing the batch size without changing any other details of the training pipeline will often affect the validation set performance.
- However, the difference in validation set performance between two batch sizes typically goes away if the training pipeline is optimized independently for each batch size.
- The hyperparameters that interact most strongly with the batch size, and therefore are most important to tune separately for each batch size, are the optimizer hyperparameters (e.g. learning rate, momentum) and the regularization hyperparameters.
	- Smaller batch sizes introduce more noise into the training algorithm due to sample variance, and this noise can have a regularizing effect. Thus, larger batch sizes can be more prone to overfitting and may require stronger regularization and/or additional regularization techniques.
- In addition, the number of training steps may need to be adjusted when changing the batch size.
- Once all these effects are taken into account, there is currently no convincing evidence that the batch size affects the maximum achievable validation performance (see Shallue et al. 2018).


为什么批量大小不应该被调整以直接提高验证集性能？

- 改变批量大小而不改变训练管道的任何其他细节通常会影响验证集性能。
- 然而，如果为每个批量大小单独优化训练管道，则两个批量大小之间的验证集性能差异通常会消失。
- 与批量大小最强烈交互的超参数，因此对于每个批量大小单独调整最重要的是优化器超参数（例如学习率、动量）和正则化超参数。
较小的批量大小由于样本方差引入更多的噪声到训练算法中，这种噪声可以产生正则化效果。因此，较大的批量大小可能更容易过拟合，可能需要更强的正则化和/或额外的正则化技术。
- 此外，当改变批量大小时，可能需要调整训练步骤的数量。
- 一旦考虑了所有这些影响，目前没有令人信服的证据表明批量大小会影响最大可达到的验证性能（参见Shallue等人2018年的研究）。

所有流行的优化算法的更新规则是什么？
Stochastic gradient descent (SGD)

\theta_{t+1}=\theta_{t}-\eta_{t} \nabla l\left(\theta_{t}\right)

Momentum

\begin{array}{c}
v_{0}=0 \\
v_{t+1}=\gamma v_{t}+\nabla l\left(\theta_{t}\right) \\
\theta_{t+1}=\theta_{t}-\eta_{t} v_{t+1}
\end{array}

Nesterov

\begin{array}{c}
v_{0}=0 \\
v_{t+1}=\gamma v_{t}+\nabla l\left(\theta_{t}\right) \\
\theta_{t+1}=\theta_{t}-\eta_{t}\left(\gamma v_{t+1}+\nabla l\left(\theta_{t}\right)\right.
\end{array}

RMSProp

\begin{array}{c}
v_{0}=1, m_{0}=0 \\
v_{t+1}=\rho v_{t}+(1-\rho) \nabla l\left(\theta_{t}\right)^{2} \\
m_{t+1}=\gamma m_{t}+\frac{\eta_{t}}{\sqrt{v_{t+1}+\epsilon}} \nabla l\left(\theta_{t}\right) \\
\theta_{t+1}=\theta_{t}-m_{t+1}
\end{array}

ADAM

\begin{array}{c}
m_{0}=0, v_{0}=0 \\
m_{t+1}=\beta_{1} m_{t}+\left(1-\beta_{1}\right) \nabla l\left(\theta_{t}\right) \\
v_{t+1}=\beta_{2} v_{t}+\left(1-\beta_{2}\right) \nabla l\left(\theta_{t}\right)^{2} \\
b_{t+1}=\frac{\sqrt{1-\beta_{2}^{t+1}}}{1-\beta_{1}^{t+1}} \\
\theta_{t+1}=\theta_{t}-\alpha_{t} \frac{m_{t+1}}{\sqrt{v_{t+1}}+\epsilon} b_{t+1}
\end{array}

NADAM

\begin{array}{c}
m_{0}=0, v_{0}=0 \\
m_{t+1}=\beta_{1} m_{t}+\left(1-\beta_{1}\right) \nabla l\left(\theta_{t}\right) \\
v_{t+1}=\beta_{2} v_{t}+\left(1-\beta_{2}\right) \nabla l\left(\theta_{t}\right)^{2} \\
b_{t+1}=\frac{\sqrt{1-\beta_{2}^{t+1}}}{1-\beta_{1}^{t+1}} \\
\theta_{t+1}=\theta_{t}-\alpha_{t} \frac{\beta_{1} m_{t+1}+\left(1-\beta_{1}\right) \nabla l\left(\theta_{t}\right)}{\sqrt{v_{t+1}}+\epsilon} b_{t+1}
\end{array}

















