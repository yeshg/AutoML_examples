# AutoML Examples

Compilation of programs for various topics in AutoML using the Ray framework from UC Berkeley's RiseLab

## Examples:

- `set_env_vars_example.py`  
-- how to start off actors in Ray Tune with different environment variables
- `pytorch_hparams.py`  
-- classic example of finding optimal hyperparameters for simple mnist classifying CNN
- `naive_pytorch_nas.py`  
-- very simple example of naive neural architecture search with simple mnist classifying CNN  
-- naive because this is effectively just another hyperparameter sweep, not actual architecture search
- `tune_resnet50.py`  
-- tuning inference time for resnet50 with imagenet weights.  
-- uses Intel Optimized tensorflow

## References

- [Ray documentation](https://docs.ray.io/en/latest/)
- [Intro to Neural Architecture Search](https://medium.com/@SmartLabAI/introduction-to-neural-architecture-search-reinforcement-learning-approach-55604772f173)