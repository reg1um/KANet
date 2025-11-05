# KANet
Kolmogorov–Arnold Networks Reimplementation

<img width="1000" height="414" alt="image" src="https://github.com/user-attachments/assets/54a2cc8b-532d-47b6-abe2-dd1e12596896" />

## ROADMAP

### Mathematical Foundations
- [X] Implement B-spline basis functions
- [X] Add unit tests for B-spline properties
- [X] Plot B-spline basis functions to verify correctness

### Core Layer
- [X] Implement basic KANLayer (forward pass only)
- [X] Add gradient flow tests (backward pass)
- [X] Integrate base activation (SiLU)
- [X] Implement grid adaptation logic
- [X] Test layer can learn simple functions (x², sin(x))

### Network Architecture
- [X] Implement KANNetwork (stack multiple layers)
- [ ] Test multi-layer forward/backward pass
- [X] Add proper weight initialization strategies
- [X] Implement grid update for entire network

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

### What's Next
- [X] Implement grid initialization and extension
- [ ] Test grid boundary handling
- [ ] Reproduce results from original KAN paper
- [ ] Profile runtime and memory
- [ ] Symbolic formula extraction
- [ ] Network pruning (remove weak connections)
- [ ] Efficient KAN optimizations
- [ ] Add more examples and tutorials
