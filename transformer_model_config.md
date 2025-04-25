# Transformer Model Configuration for GB200 Superchip

## Model Size

- **Number of Parameters**: 824,000,000,000 (824 billion)

## Dimensions and Layers

- **Number of Layers (( L ))**: 256
- **Hidden Dimension (( d\_{\\text{model}} ))**: 16,384

## Other Hyperparameters

- **Number of Attention Heads**: 128
- **Feedforward Dimension (( d\_{\\text{ff}} ))**: 65,536
- **Batch Size × Sequence Length (( B \\times S ))**: &gt; 658 (e.g., ( B = 1 ), ( S = 1024 ))

## Notes

- Precision: FP4 (0.5 bytes/parameter)
- Memory: Fits within 384 GB HBM3e with optimization
- Compute-bottlenecked: Arithmetic intensity &gt; 2500 ops/byte