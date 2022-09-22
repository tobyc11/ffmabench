# ffmabench
CUDA benchmark that shows your GPU's TFLOPS

# Example Output

This is on a GeForce RTX 2060 SUPER. It is 7.2 TFLOPs on paper.

```
Running 16000 ops/thread * 1024 threads/block * 1048576 blocks
Launch latency: 0.000040 s
Kernel duration: 2.100975 s
Your GPU's TFLOPS is 8.177092
```

And this is on a Quadro GP100. 10.3 TFLOPs on paper.

```
Running 16000 ops/thread * 1024 threads/block * 1048576 blocks
Launch latency: 0.000051 s
Kernel duration: 1.556323 s
Your GPU's TFLOPS is 11.038755
```
