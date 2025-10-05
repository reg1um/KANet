# KANet
Kolmogorov–Arnold Networks Reimplementation

## ROADMAP

### Mathematical Foundations
- [ ] Implement B-spline basis functions
- [ ] Add unit tests for B-spline properties
- [ ] Implement grid initialization and extension
- [ ] Test grid boundary handling
- [ ] Plot B-spline basis functions to verify correctness

### Core Layer
- [ ] Implement basic KANLayer (forward pass only)
- [ ] Add gradient flow tests (backward pass)
- [ ] Integrate base activation (SiLU)
- [ ] Implement grid adaptation logic
- [ ] Test layer can learn simple functions (x², sin(x))

### Network Architecture
- [ ] Implement KANNetwork (stack multiple layers)
- [ ] Test multi-layer forward/backward pass
- [ ] Add proper weight initialization strategies
- [ ] Implement grid update for entire network

### Training Infrastructure
- [ ] Create KANTrainer with basic training loop
- [ ] Add grid update scheduling (every N epochs)
- [ ] Implement L1 regularization on coefficients
- [ ] Add learning rate scheduling

### Validation & Tools
- [ ] Create visualization tools (plot learned functions)
- [ ] Implement regression example (compare vs MLP)
- [ ] Add classification example
- [ ] Benchmark on standard datasets
- [ ] Document usage and API

### What's Next (needs more research...)
- [ ] Reproduce results from original KAN paper
- [ ] Profile runtime and memory
- [ ] Symbolic formula extraction
- [ ] Network pruning (remove weak connections)
- [ ] Efficient KAN optimizations
- [ ] Add more examples and tutorials
